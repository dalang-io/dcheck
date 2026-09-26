//! `dcheck undelete`: list and recover deleted files.
//!
//! Works on a partition, a whole disk (MBR / GPT partitions are found) or an
//! image made with ddrescue — always opened read-only.
//!
//! - NTFS: a deleted file's MFT record keeps its name, parent folder, size
//!   and data runs (fragmented files too) until the record is reused; small
//!   files live inside the record itself. When a file is so fragmented that
//!   its runs no longer fit in one record, the base record points at
//!   extension records through `$ATTRIBUTE_LIST`; those are followed too, and
//!   the runlist of every extent is decoded on its own (each extent's first
//!   LCN is absolute). Windows and ntfs-3g keep the name; the Linux ntfs3
//!   driver removes it from the MFT record, so the name is looked for in the
//!   directory index (`$I30`) slack, and only when that fails is the file
//!   recovered as `$NoName/record-N.<type from its contents>`.
//! - FAT32: the first byte of the name becomes 0xE5 and the cluster chain is
//!   cleared, but the long name, size and first cluster remain. **The data is
//!   assumed to be contiguous** from the first cluster: correct for the vast
//!   majority of unfragmented files, wrong for a fragmented one (its tail is
//!   then taken from whatever clusters follow). Such files are marked
//!   "assumed contiguous".
//! - exFAT: the InUse bit is cleared; name, size, first cluster and the
//!   NoFatChain ("contiguous") flag remain. A file that was fragmented has
//!   NoFatChain clear and its FAT chain is gone, so it too falls back to the
//!   same contiguous assumption (with the same note).
//! - Anything else (ext4, XFS, btrfs …): the block map is gone, so
//!   `--carve` searches the raw device for JPEG / PNG / PDF / ZIP (Office)
//!   files instead. Names are lost. `--carve --free` restricts the search to
//!   the free clusters of a filesystem whose bitmap is known (NTFS / FAT32 /
//!   exFAT), so live files are not carved again.
//!
//! Every file gets a state from the allocation table / bitmap: its clusters
//! are still free (INTACT), some are in use again (PARTLY REUSED), or all of
//! them are (OVERWRITTEN).

#![cfg_attr(not(target_os = "linux"), allow(dead_code))]
// Fixed-size on-disk records (FAT entries, directory entries, UTF-16 units)
// read clearer as `chunks_exact(n)` than as `as_chunks::<n>()`.
#![allow(clippy::chunks_exact_to_as_chunks)]

use std::collections::{HashMap, HashSet};
use std::io;

use crate::report::human_size_bin;

/// Random-access, read-only bytes: a device, an image, or a test buffer.
pub trait Source {
    fn read_at(&self, off: u64, buf: &mut [u8]) -> io::Result<()>;
    fn len(&self) -> u64;
}

impl Source for Vec<u8> {
    fn read_at(&self, off: u64, buf: &mut [u8]) -> io::Result<()> {
        let start = off as usize;
        let end = start + buf.len();
        if end > self.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "past the end"));
        }
        buf.copy_from_slice(&self[start..end]);
        Ok(())
    }
    fn len(&self) -> u64 {
        Vec::len(self) as u64
    }
}

#[cfg(unix)]
pub struct FileSource {
    file: std::fs::File,
    len: u64,
}

#[cfg(unix)]
impl FileSource {
    pub fn open(path: &str) -> io::Result<FileSource> {
        use std::io::{Seek, SeekFrom};
        let mut file = std::fs::File::open(path)?; // read-only
        let len = file.seek(SeekFrom::End(0))?;
        Ok(FileSource { file, len })
    }
}

#[cfg(unix)]
impl Source for FileSource {
    fn read_at(&self, off: u64, buf: &mut [u8]) -> io::Result<()> {
        std::os::unix::fs::FileExt::read_exact_at(&self.file, buf, off)
    }
    fn len(&self) -> u64 {
        self.len
    }
}

fn read(src: &dyn Source, off: u64, len: usize) -> io::Result<Vec<u8>> {
    let mut b = vec![0u8; len];
    src.read_at(off, &mut b)?;
    Ok(b)
}

fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u64le(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

fn utf16(b: &[u8]) -> String {
    let units: Vec<u16> = b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    String::from_utf16_lossy(&units)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Every cluster is still free.
    Intact,
    /// Some clusters are in use by another file now.
    PartlyReused,
    /// All clusters are in use again: the contents are gone.
    Overwritten,
}

impl State {
    pub fn label(self) -> &'static str {
        match self {
            State::Intact => "INTACT",
            State::PartlyReused => "PARTLY REUSED",
            State::Overwritten => "OVERWRITTEN",
        }
    }
}

/// Where a deleted file's bytes are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Data {
    /// (byte offset in the source, length); `None` offset = sparse (zeros).
    Extents(Vec<(Option<u64>, u64)>),
    /// Stored inside the NTFS record.
    Resident(Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct Deleted {
    /// Partition label, e.g. "sdb1 (NTFS)" or "whole device (FAT32)".
    pub volume: String,
    pub path: String,
    pub size: u64,
    pub state: State,
    pub data: Data,
    /// How sure the location is ("assumed contiguous").
    pub note: Option<&'static str>,
}

/// A filesystem found on the source.
#[derive(Debug, Clone)]
pub struct Volume {
    pub label: String,
    pub fs: &'static str,
    pub base: u64,
    pub size: u64,
}

/// Cells of the volume map (fraction of allocated clusters per cell).
pub const MAP_CELLS: usize = 1024;

/// Where a volume's space is in use now, for the block map.
#[derive(Debug, Clone)]
pub struct VolMap {
    pub volume: String,
    /// Byte range of the volume in the source.
    pub base: u64,
    pub size: u64,
    /// `MAP_CELLS` values in 0.0..=1.0: share of allocated clusters.
    pub used: Vec<f32>,
}

/// Build a map from "is cluster N allocated" over the cluster heap
/// (`heap` = byte offset of cluster `first` in the volume).
fn vol_map(vol: &Volume, fs: &str, heap: u64, cs: u64, first: u64, count: u64, used: impl Fn(u64) -> bool) -> VolMap {
    let mut sum = vec![0f64; MAP_CELLS];
    let mut n = vec![0f64; MAP_CELLS];
    let step = (count / (MAP_CELLS as u64 * 256)).max(1); // sample big volumes
    let mut c = 0;
    while c < count {
        let byte = heap + c * cs;
        let cell = ((byte as u128 * MAP_CELLS as u128 / vol.size.max(1) as u128) as usize).min(MAP_CELLS - 1);
        n[cell] += 1.0;
        if used(first + c) {
            sum[cell] += 1.0;
        }
        c += step;
    }
    // Metadata before the heap (boot sector, FAT) counts as used.
    let meta_cells = ((heap as u128 * MAP_CELLS as u128 / vol.size.max(1) as u128) as usize).min(MAP_CELLS);
    let used = (0..MAP_CELLS)
        .map(|i| if i < meta_cells { 1.0 } else if n[i] > 0.0 { (sum[i] / n[i]) as f32 } else { 0.0 })
        .collect();
    VolMap { volume: format!("{} ({fs})", vol.label), base: vol.base, size: vol.size, used }
}

// ─── partitions ─────────────────────────────────────────────────────────────

fn fs_at(src: &dyn Source, base: u64) -> Option<&'static str> {
    let b = read(src, base, 512).ok()?;
    if &b[3..11] == b"NTFS    " {
        Some("NTFS")
    } else if &b[3..11] == b"EXFAT   " {
        Some("exFAT")
    } else if &b[82..90] == b"FAT32   " {
        Some("FAT32")
    } else {
        let sb = read(src, base + 1024, 64).ok()?;
        if u16le(&sb, 56) == 0xEF53 {
            Some("ext4")
        } else if read(src, base, 4).ok()? == b"XFSB" {
            Some("XFS")
        } else if read(src, base + 0x10040, 8).ok()? == b"_BHRfS_M" {
            Some("btrfs")
        } else {
            None
        }
    }
}

/// Filesystems on the source: the device itself, or its MBR / GPT partitions.
pub fn volumes(src: &dyn Source) -> Vec<Volume> {
    if let Some(fs) = fs_at(src, 0) {
        return vec![Volume { label: "whole device".into(), fs, base: 0, size: src.len() }];
    }
    let mut parts: Vec<(u64, u64)> = Vec::new();
    if let Ok(gpt) = read(src, 512, 512) {
        if &gpt[..8] == b"EFI PART" {
            let lba = u64le(&gpt, 72);
            let n = u32le(&gpt, 80).min(256) as usize;
            let sz = u32le(&gpt, 84) as usize;
            if let Ok(t) = read(src, lba * 512, n * sz) {
                for e in t.chunks(sz) {
                    if e[..16].iter().any(|b| *b != 0) {
                        let (first, last) = (u64le(e, 32), u64le(e, 40));
                        parts.push((first * 512, (last + 1 - first) * 512));
                    }
                }
            }
        }
    }
    if parts.is_empty() {
        if let Ok(mbr) = read(src, 0, 512) {
            if mbr[510] == 0x55 && mbr[511] == 0xAA {
                for i in 0..4 {
                    let e = &mbr[446 + i * 16..446 + (i + 1) * 16];
                    let (kind, start, len) = (e[4], u32le(e, 8) as u64, u32le(e, 12) as u64);
                    if kind != 0 && kind != 0xEE && kind != 0x05 && kind != 0x0F && len > 0 {
                        parts.push((start * 512, len * 512));
                    }
                }
            }
        }
    }
    parts
        .into_iter()
        .enumerate()
        .filter_map(|(i, (base, size))| {
            fs_at(src, base).map(|fs| Volume { label: format!("partition {}", i + 1), fs, base, size })
        })
        .collect()
}

// ─── FAT32 ──────────────────────────────────────────────────────────────────

struct Fat32<'a> {
    src: &'a dyn Source,
    base: u64,
    cs: u64,
    data: u64,
    fat: Vec<u32>,
}

impl Fat32<'_> {
    fn cluster_off(&self, c: u32) -> u64 {
        self.base + self.data + (c as u64 - 2) * self.cs
    }

    fn valid(&self, c: u32) -> bool {
        c >= 2 && (c as usize) < self.fat.len()
    }

    /// Live chain from `c` (bounded against loops).
    fn chain(&self, mut c: u32) -> Vec<u32> {
        let mut v = Vec::new();
        let mut seen = HashSet::new();
        while self.valid(c) && seen.insert(c) {
            v.push(c);
            c = self.fat[c as usize] & 0x0FFF_FFFF;
            if c >= 0x0FFF_FFF8 {
                break;
            }
        }
        v
    }

    fn read_clusters(&self, cl: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &c in cl {
            if let Ok(b) = read(self.src, self.cluster_off(c), self.cs as usize) {
                out.extend(b);
            }
        }
        out
    }
}

fn sfn(e: &[u8]) -> String {
    let base = String::from_utf8_lossy(&e[..8]).trim_end().to_string();
    let ext = String::from_utf8_lossy(&e[8..11]).trim_end().to_string();
    if ext.is_empty() {
        base
    } else {
        format!("{base}.{ext}")
    }
}

fn lfn_part(e: &[u8]) -> Vec<u16> {
    let mut v = Vec::new();
    for (o, n) in [(1usize, 5usize), (14, 6), (28, 2)] {
        for k in 0..n {
            v.push(u16le(e, o + 2 * k));
        }
    }
    v
}

fn fat32(src: &dyn Source, vol: &Volume) -> io::Result<(Vec<Deleted>, VolMap)> {
    let b = read(src, vol.base, 512)?;
    let bps = u16le(&b, 11) as u64;
    let spc = b[13] as u64;
    let reserved = u16le(&b, 14) as u64;
    let nfats = b[16] as u64;
    let fatsz = u32le(&b, 36) as u64;
    let root = u32le(&b, 44);
    if bps == 0 || spc == 0 || fatsz == 0 {
        return Err(io::Error::other("bad FAT32 boot sector"));
    }
    let raw = read(src, vol.base + reserved * bps, (fatsz * bps) as usize)?;
    let fat: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let f = Fat32 { src, base: vol.base, cs: bps * spc, data: (reserved + nfats * fatsz) * bps, fat };
    let count = (vol.size.saturating_sub(f.data) / f.cs).min(f.fat.len() as u64 - 2);
    let map = vol_map(vol, "FAT32", f.data, f.cs, 2, count, |c| f.fat.get(c as usize).is_some_and(|v| v & 0x0FFF_FFFF != 0));
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    // (first cluster, path, directory is deleted)
    let mut stack = vec![(root, String::new(), false)];
    while let Some((dir, prefix, dir_deleted)) = stack.pop() {
        if !seen.insert(dir) || !f.valid(dir) {
            continue;
        }
        let clusters = if dir_deleted {
            // Chain cleared: read contiguous clusters (directories are small).
            (dir..dir.saturating_add(8)).filter(|c| f.valid(*c)).collect()
        } else {
            f.chain(dir)
        };
        let bytes = f.read_clusters(&clusters);
        let mut lfn: Vec<Vec<u16>> = Vec::new();
        for e in bytes.chunks_exact(32) {
            if e[0] == 0 {
                break;
            }
            if e[11] == 0x0F {
                lfn.push(lfn_part(e));
                continue;
            }
            let parts = std::mem::take(&mut lfn);
            if e[11] & 0x08 != 0 || e[0] == b'.' {
                continue;
            }
            let deleted = e[0] == 0xE5;
            let long: Vec<u16> = parts.iter().rev().flatten().copied().take_while(|u| *u != 0 && *u != 0xFFFF).collect();
            let name = if !long.is_empty() {
                String::from_utf16_lossy(&long)
            } else {
                let mut n = sfn(e);
                if deleted {
                    n.replace_range(..1, "_");
                }
                n
            };
            let first = ((u16le(e, 20) as u32) << 16) | u16le(e, 26) as u32;
            let size = u32le(e, 28) as u64;
            let path = format!("{prefix}{name}");
            if e[11] & 0x10 != 0 {
                if first >= 2 {
                    stack.push((first, format!("{path}/"), deleted || dir_deleted));
                }
                continue;
            }
            if !(deleted || dir_deleted) || !f.valid(first) {
                continue;
            }
            let n = size.div_ceil(f.cs).max(1);
            let clusters: Vec<u32> = (first..first.saturating_add(n as u32)).filter(|c| f.valid(*c)).collect();
            let used = clusters.iter().filter(|c| f.fat[**c as usize] & 0x0FFF_FFFF != 0).count();
            out.push(Deleted {
                volume: format!("{} (FAT32)", vol.label),
                path,
                size,
                state: state(used, clusters.len()),
                data: Data::Extents(vec![(Some(f.cluster_off(first)), size)]),
                note: (n > 1).then_some("assumed contiguous"),
            });
        }
    }
    Ok((out, map))
}

fn state(used: usize, total: usize) -> State {
    match used {
        0 => State::Intact,
        u if u >= total => State::Overwritten,
        _ => State::PartlyReused,
    }
}

// ─── exFAT ──────────────────────────────────────────────────────────────────

fn exfat(src: &dyn Source, vol: &Volume) -> io::Result<(Vec<Deleted>, VolMap)> {
    let b = read(src, vol.base, 512)?;
    let bps = 1u64 << b[108];
    let cs = bps << b[109];
    let fat_off = u32le(&b, 80) as u64 * bps;
    let heap = u32le(&b, 88) as u64 * bps;
    let count = u32le(&b, 92);
    let root = u32le(&b, 96);
    let fat_raw = read(src, vol.base + fat_off, (count as usize + 2) * 4)?;
    let fat: Vec<u32> = fat_raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let off = |c: u32| vol.base + heap + (c as u64 - 2) * cs;
    let valid = |c: u32| c >= 2 && c < count + 2;
    let chain = |mut c: u32| {
        let mut v = Vec::new();
        let mut seen = HashSet::new();
        while valid(c) && seen.insert(c) {
            v.push(c);
            c = fat[c as usize];
            if c >= 0xFFFF_FFF7 {
                break;
            }
        }
        v
    };
    let read_cl = |cl: &[u32]| {
        let mut o = Vec::new();
        for &c in cl {
            if let Ok(x) = read(src, off(c), cs as usize) {
                o.extend(x);
            }
        }
        o
    };
    // Allocation bitmap (root entry 0x81).
    let mut bitmap = Vec::new();
    for e in read_cl(&chain(root)).chunks_exact(32) {
        if e[0] == 0x81 {
            let (c, len) = (u32le(e, 20), u64le(e, 24));
            let mut bm = read_cl(&chain(c));
            bm.truncate(len as usize);
            bitmap = bm;
            break;
        }
    }
    let in_use = |c: u32| {
        let i = (c - 2) as usize;
        bitmap.get(i / 8).is_some_and(|b| b & (1 << (i % 8)) != 0)
    };
    let map = vol_map(vol, "exFAT", heap, cs, 2, count as u64, |c| in_use(c as u32));
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    // (cluster, contiguous length in bytes or 0 = follow FAT, path, deleted)
    let mut stack = vec![(root, 0u64, String::new(), false)];
    while let Some((dir, len, prefix, dir_deleted)) = stack.pop() {
        if !seen.insert(dir) || !valid(dir) {
            continue;
        }
        let clusters: Vec<u32> = if len > 0 {
            (dir..dir.saturating_add(len.div_ceil(cs) as u32)).filter(|c| valid(*c)).collect()
        } else {
            chain(dir)
        };
        let bytes = read_cl(&clusters);
        let ents: Vec<&[u8]> = bytes.chunks_exact(32).collect();
        let mut i = 0;
        while i < ents.len() {
            let e = ents[i];
            if e[0] == 0 {
                break;
            }
            if e[0] & 0x7F != 0x05 {
                i += 1;
                continue;
            }
            let deleted = e[0] & 0x80 == 0;
            let secondary = e[1] as usize;
            let is_dir = u16le(e, 4) & 0x10 != 0;
            if i + secondary >= ents.len() || secondary < 2 {
                i += 1;
                continue;
            }
            let st = ents[i + 1];
            if st[0] & 0x7F != 0x40 {
                i += 1;
                continue;
            }
            let contiguous = st[1] & 0x02 != 0;
            let name_len = st[3] as usize;
            let (first, size) = (u32le(st, 20), u64le(st, 24));
            let mut name_u: Vec<u16> = Vec::new();
            for n in &ents[i + 2..=i + secondary] {
                if n[0] & 0x7F == 0x41 {
                    name_u.extend(n[2..32].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])));
                }
            }
            name_u.truncate(name_len);
            let path = format!("{prefix}{}", String::from_utf16_lossy(&name_u));
            i += secondary + 1;
            if is_dir {
                if valid(first) {
                    stack.push((first, if contiguous || deleted { size } else { 0 }, format!("{path}/"), deleted || dir_deleted));
                }
                continue;
            }
            if !(deleted || dir_deleted) || size == 0 || !valid(first) {
                continue;
            }
            let n = size.div_ceil(cs) as usize;
            let (clusters, note) = if contiguous {
                ((first..first + n as u32).filter(|c| valid(*c)).collect::<Vec<_>>(), None)
            } else {
                let c = chain(first);
                if c.len() >= n {
                    (c[..n].to_vec(), None)
                } else {
                    ((first..first + n as u32).filter(|c| valid(*c)).collect(), Some("assumed contiguous"))
                }
            };
            let used = clusters.iter().filter(|c| in_use(**c)).count();
            let mut ext: Vec<(Option<u64>, u64)> = Vec::new();
            let mut left = size;
            for c in &clusters {
                let take = left.min(cs);
                match ext.last_mut() {
                    Some((Some(o), l)) if *o + *l == off(*c) => *l += take,
                    _ => ext.push((Some(off(*c)), take)),
                }
                left -= take;
            }
            out.push(Deleted {
                volume: format!("{} (exFAT)", vol.label),
                path,
                size,
                state: state(used, clusters.len()),
                data: Data::Extents(ext),
                note,
            });
        }
    }
    Ok((out, map))
}

// ─── NTFS ───────────────────────────────────────────────────────────────────

/// Apply the update sequence array of an MFT record (in place).
pub fn ntfs_fixup(rec: &mut [u8]) -> bool {
    fixup(rec, b"FILE")
}

/// Data runs: (LCN or None for sparse, cluster count).
pub fn ntfs_runs(rl: &[u8]) -> Vec<(Option<i64>, u64)> {
    let mut out = Vec::new();
    let (mut i, mut lcn) = (0usize, 0i64);
    while i < rl.len() && rl[i] != 0 {
        let (ln, lo) = ((rl[i] & 0x0F) as usize, (rl[i] >> 4) as usize);
        i += 1;
        if i + ln + lo > rl.len() || ln == 0 || ln > 8 || lo > 8 {
            break;
        }
        let mut cnt = 0u64;
        for k in 0..ln {
            cnt |= (rl[i + k] as u64) << (8 * k);
        }
        i += ln;
        if lo == 0 {
            out.push((None, cnt));
            continue;
        }
        let mut v = 0i64;
        for k in 0..lo {
            v |= (rl[i + k] as i64) << (8 * k);
        }
        if rl[i + lo - 1] & 0x80 != 0 && lo < 8 {
            v -= 1i64 << (8 * lo);
        }
        i += lo;
        lcn += v;
        out.push((Some(lcn), cnt));
    }
    out
}

/// One attribute of an MFT record, as far as undelete needs it.
#[derive(Debug, Clone)]
struct NtfsAttr {
    type_: u32,
    /// The attribute has a name (e.g. `$I30`); unnamed `$DATA`/`$FILE_NAME` do not.
    named: bool,
    /// Lowest VCN of a non-resident attribute (0 for its first extent).
    vcn: u64,
    /// Real data size; only the first extent carries it.
    size: u64,
    /// Resident value bytes.
    resident: Option<Vec<u8>>,
    /// Raw mapping-pairs runlist of a non-resident attribute.
    runs: Option<Vec<u8>>,
}

/// One entry of a record's `$ATTRIBUTE_LIST`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AttrRef {
    type_: u32,
    #[allow(dead_code)]
    id: u16,
    named: bool,
    /// MFT record number holding this attribute extent.
    record: u64,
    vcn: u64,
}

struct NtfsRecord {
    in_use: bool,
    is_dir: bool,
    attrs: Vec<NtfsAttr>,
}

impl NtfsRecord {
    /// Best `$FILE_NAME` value: (parent record, name). The Win32 namespace is
    /// preferred over the 8.3 DOS one.
    fn name(&self) -> Option<(u64, String)> {
        let mut best: Option<(u8, u64, String)> = None;
        for a in &self.attrs {
            if a.type_ != 0x30 {
                continue;
            }
            let Some(c) = &a.resident else { continue };
            if c.len() < 66 {
                continue;
            }
            let (n, ns) = (c[64] as usize, c[65]);
            if 66 + 2 * n > c.len() {
                continue;
            }
            let rank = if ns == 2 { 3 } else { ns };
            if best.as_ref().is_none_or(|(r, _, _)| rank < *r) {
                best = Some((rank, u64le(c, 0) & 0x0000_FFFF_FFFF_FFFF, utf16(&c[66..66 + 2 * n])));
            }
        }
        best.map(|(_, p, n)| (p, n))
    }

    fn has_data(&self) -> bool {
        self.attrs.iter().any(|a| a.type_ == 0x80 && !a.named)
    }
}

fn ntfs_record(rec: &[u8]) -> NtfsRecord {
    let flags = u16le(rec, 22);
    let mut r = NtfsRecord { in_use: flags & 1 != 0, is_dir: flags & 2 != 0, attrs: Vec::new() };
    let mut off = u16le(rec, 20) as usize;
    while off + 16 <= rec.len() {
        let t = u32le(rec, off);
        let len = u32le(rec, off + 4) as usize;
        if t == 0xFFFF_FFFF || len < 16 || off + len > rec.len() {
            break;
        }
        let a = &rec[off..off + len];
        let nonres = a[8] != 0;
        let named = a[9] != 0;
        let mut attr = NtfsAttr { type_: t, named, vcn: 0, size: 0, resident: None, runs: None };
        if nonres {
            attr.vcn = u64le(a, 16);
            attr.size = u64le(a, 48);
            let ro = u16le(a, 32) as usize;
            if ro < a.len() {
                attr.runs = Some(a[ro..].to_vec());
            }
        } else {
            let (co, cl) = (u16le(a, 20) as usize, u32le(a, 16) as usize);
            if co + cl <= a.len() {
                attr.resident = Some(a[co..co + cl].to_vec());
            }
        }
        r.attrs.push(attr);
        off += len;
    }
    r
}

/// Parse the entries of a `$ATTRIBUTE_LIST` value.
fn attr_list_entries(b: &[u8]) -> Vec<AttrRef> {
    let mut v = Vec::new();
    let mut off = 0usize;
    while off + 26 <= b.len() {
        let t = u32le(b, off);
        if t == 0xFFFF_FFFF {
            break;
        }
        let len = u16le(b, off + 4) as usize;
        if len < 26 || off + len > b.len() {
            break;
        }
        v.push(AttrRef {
            type_: t,
            id: u16le(b, off + 24),
            named: b[off + 6] != 0,
            record: u64le(b, off + 16) & 0x0000_FFFF_FFFF_FFFF,
            vcn: u64le(b, off + 8),
        });
        off += len;
    }
    v
}

/// Read the clusters of a decoded runlist into a buffer of at most `size` bytes.
fn read_runs(src: &dyn Source, runs: &[(Option<i64>, u64)], size: u64, lcn_off: &dyn Fn(i64) -> u64, cs: u64) -> Vec<u8> {
    let mut v = Vec::new();
    for (l, c) in runs {
        match l {
            Some(l) => v.extend(read(src, lcn_off(*l), (*c * cs) as usize).unwrap_or_default()),
            None => v.extend(vec![0u8; (*c * cs) as usize]),
        }
    }
    v.truncate(size as usize);
    v
}

/// `$ATTRIBUTE_LIST` entries of a record, reading the list from disk when it is
/// itself non-resident.
fn record_attr_list(recs: &HashMap<u64, NtfsRecord>, i: u64, src: &dyn Source, lcn_off: &dyn Fn(i64) -> u64, cs: u64) -> Vec<AttrRef> {
    let Some(r) = recs.get(&i) else { return Vec::new() };
    let mut v = Vec::new();
    for a in &r.attrs {
        if a.type_ != 0x20 {
            continue;
        }
        if let Some(b) = &a.resident {
            v.extend(attr_list_entries(b));
        } else if let Some(raw) = &a.runs {
            let bytes = read_runs(src, &ntfs_runs(raw), a.size, lcn_off, cs);
            v.extend(attr_list_entries(&bytes));
        }
    }
    v
}

/// A resolved attribute value: resident bytes, or a runlist plus its real size.
#[derive(Debug)]
enum Resolved {
    Resident(Vec<u8>),
    Runs(Vec<(Option<i64>, u64)>, u64),
}

/// Gather the extents of an attribute from the base record and the extension
/// records named in its `$ATTRIBUTE_LIST`, and merge them in VCN order.
///
/// Every extent's runlist is decoded on its own: each one starts with an
/// absolute LCN (the first mapping-pair offset is relative to 0), so no
/// correction between extents is needed.
fn resolve_attr(recs: &HashMap<u64, NtfsRecord>, list: &[AttrRef], base: u64, type_: u32, unnamed: bool) -> Option<Resolved> {
    let mut segs: Vec<(u64, NtfsAttr)> = Vec::new();
    let mut collect = |rec: Option<&NtfsRecord>| {
        if let Some(r) = rec {
            for a in &r.attrs {
                if a.type_ == type_ && (!unnamed || !a.named) {
                    segs.push((a.vcn, a.clone()));
                }
            }
        }
    };
    collect(recs.get(&base));
    for e in list {
        if e.type_ == type_ && (!unnamed || !e.named) {
            collect(recs.get(&e.record));
        }
    }
    if segs.is_empty() {
        return None;
    }
    if let Some((_, a)) = segs.iter().find(|(_, a)| a.resident.is_some()) {
        return Some(Resolved::Resident(a.resident.clone().unwrap()));
    }
    segs.sort_by_key(|(v, _)| *v);
    segs.dedup_by_key(|(v, _)| *v);
    let mut runs = Vec::new();
    let mut size = 0u64;
    for (vcn, a) in &segs {
        runs.extend(ntfs_runs(a.runs.as_deref().unwrap_or(&[])));
        if *vcn == 0 {
            size = a.size;
        }
    }
    if size == 0 {
        size = segs.iter().map(|(_, a)| a.size).max().unwrap_or(0);
    }
    Some(Resolved::Runs(runs, size))
}

// ─── NTFS directory index (names for records whose $FILE_NAME is gone) ───────

/// Fix up a record guarded by `magic` (`FILE` or `INDX`).
fn fixup(rec: &mut [u8], magic: &[u8; 4]) -> bool {
    if rec.len() < 48 || &rec[..4] != magic {
        return false;
    }
    let (uo, un) = (u16le(rec, 4) as usize, u16le(rec, 6) as usize);
    if uo + 2 * un > rec.len() {
        return false;
    }
    let usn = [rec[uo], rec[uo + 1]];
    for i in 1..un {
        let p = i * 512 - 2;
        if p + 2 > rec.len() || rec[p..p + 2] != usn {
            return false;
        }
        rec[p] = rec[uo + 2 * i];
        rec[p + 1] = rec[uo + 2 * i + 1];
    }
    true
}

/// Walk `$FILE_NAME` index entries from `entries_off` (relative to the buffer)
/// until the last-entry flag, calling `f(record, parent, name)`.
fn index_entries(buf: &[u8], entries_off: usize, mut f: impl FnMut(u64, u64, String)) {
    let mut p = entries_off;
    while p + 16 <= buf.len() {
        let rec = u64le(buf, p) & 0x0000_FFFF_FFFF_FFFF;
        let len = u16le(buf, p + 8) as usize;
        let stream_len = u16le(buf, p + 10) as usize;
        let flags = u32le(buf, p + 12);
        if len < 16 || p + len > buf.len() {
            break;
        }
        if flags & 0x02 != 0 {
            break; // last entry
        }
        let s = &buf[p + 16..(p + 16 + stream_len.min(len - 16)).min(buf.len())];
        if s.len() >= 66 {
            let parent = u64le(s, 0) & 0x0000_FFFF_FFFF_FFFF;
            let n = s[64] as usize;
            if 66 + 2 * n <= s.len() {
                f(rec, parent, utf16(&s[66..66 + 2 * n]));
            }
        }
        p += len;
    }
}

/// Entries of a resident `$INDEX_ROOT` value, if it indexes `$FILE_NAME`.
fn index_root_entries(root: &[u8]) -> Vec<(u64, u64, String)> {
    let mut out = Vec::new();
    if root.len() < 0x10 + 16 || u32le(root, 0) != 0x30 {
        return out;
    }
    let entries = 0x10 + u32le(root, 0x10) as usize;
    index_entries(root, entries, |r, p, n| out.push((r, p, n)));
    out
}

/// Entries of an `$INDEX_ALLOCATION` value: a series of `INDX` blocks.
fn index_alloc_entries(data: &[u8], block: usize) -> Vec<(u64, u64, String)> {
    let mut out = Vec::new();
    if block < 512 {
        return out;
    }
    for b in data.chunks(block) {
        if b.len() < 0x18 + 16 {
            break;
        }
        let mut b = b.to_vec();
        if !fixup(&mut b, b"INDX") {
            continue;
        }
        let entries = 0x18 + u32le(&b, 0x18) as usize;
        index_entries(&b, entries, |r, p, n| out.push((r, p, n)));
    }
    out
}

/// Map every deleted MFT record to the name the directory index still holds
/// for it (the Linux ntfs3 driver removes `$FILE_NAME` from the record but the
/// stale index entry can survive in the index buffer slack).
fn index_name_map(recs: &HashMap<u64, NtfsRecord>, src: &dyn Source, lcn_off: &dyn Fn(i64) -> u64, cs: u64) -> HashMap<u64, (u64, String)> {
    let mut out = HashMap::new();
    for i in recs.keys() {
        if !r_is_dir(recs, *i) {
            continue;
        }
        let list = record_attr_list(recs, *i, src, lcn_off, cs);
        let root = resolve_attr(recs, &list, *i, 0x90, false);
        let block = match &root {
            Some(Resolved::Resident(b)) if b.len() >= 0x0C => u32le(b, 8) as usize,
            _ => 0,
        };
        if let Some(Resolved::Resident(b)) = root {
            for (r, p, n) in index_root_entries(&b) {
                out.entry(r).or_insert((p, n));
            }
        }
        if let Some(Resolved::Runs(runs, size)) = resolve_attr(recs, &list, *i, 0xA0, false) {
            let data = read_runs(src, &runs, size, lcn_off, cs);
            for (r, p, n) in index_alloc_entries(&data, block) {
                out.entry(r).or_insert((p, n));
            }
        }
    }
    out
}

fn ntfs(src: &dyn Source, vol: &Volume) -> io::Result<(Vec<Deleted>, VolMap)> {
    let b = read(src, vol.base, 512)?;
    let bps = u16le(&b, 11) as u64;
    let cs = bps * b[13] as u64;
    let mft_lcn = u64le(&b, 48);
    let cpr = b[64] as i8;
    let rs = if cpr > 0 { cs * cpr as u64 } else { 1u64 << (-cpr as u32) };
    if cs == 0 || rs == 0 || rs > 65536 {
        return Err(io::Error::other("bad NTFS boot sector"));
    }
    let lcn_off = |l: i64| vol.base + l as u64 * cs;
    let mut rec0 = read(src, lcn_off(mft_lcn as i64), rs as usize)?;
    if !ntfs_fixup(&mut rec0) {
        return Err(io::Error::other("MFT record 0 unreadable"));
    }
    let rec0 = ntfs_record(&rec0);
    let mft_runs = rec0
        .attrs
        .iter()
        .find(|a| a.type_ == 0x80 && !a.named)
        .and_then(|a| a.runs.as_deref())
        .map(ntfs_runs);
    let Some(mft_runs) = mft_runs else {
        return Err(io::Error::other("no $MFT data runs"));
    };
    // Every record: index → parsed record.
    let mut recs: HashMap<u64, NtfsRecord> = HashMap::new();
    let mut idx = 0u64;
    for (lcn, cnt) in &mft_runs {
        let per = cnt * cs / rs;
        let Some(lcn) = lcn else {
            idx += per;
            continue;
        };
        // Read the run in 1 MiB pieces.
        let total = cnt * cs;
        let mut pos = 0u64;
        while pos < total {
            let n = (total - pos).min(1 << 20);
            let chunk = read(src, lcn_off(*lcn) + pos, n as usize)?;
            for r in chunk.chunks(rs as usize) {
                let mut r = r.to_vec();
                if ntfs_fixup(&mut r) {
                    recs.insert(idx, ntfs_record(&r));
                }
                idx += 1;
            }
            pos += n;
        }
    }
    // $Bitmap (record 6): cluster allocation now.
    let bitmap: Vec<u8> = {
        let list = record_attr_list(&recs, 6, src, &lcn_off, cs);
        match resolve_attr(&recs, &list, 6, 0x80, true) {
            Some(Resolved::Resident(v)) => v,
            Some(Resolved::Runs(runs, size)) => read_runs(src, &runs, size, &lcn_off, cs),
            None => Vec::new(),
        }
    };
    let allocated = |c: u64| bitmap.get((c / 8) as usize).is_some_and(|b| b & (1 << (c % 8)) != 0);
    let map = vol_map(vol, "NTFS", 0, cs, 0, vol.size / cs, allocated);
    // Names the directory index still holds for records whose $FILE_NAME is
    // gone (ntfs3). Only computed when a deleted file actually needs it.
    let need_names = recs.iter().any(|(i, r)| *i >= 24 && !r.in_use && !r.is_dir && r.has_data() && r.name().is_none());
    let index_names = if need_names { index_name_map(&recs, src, &lcn_off, cs) } else { HashMap::new() };
    let path_of = |mut parent: u64, name: &str| {
        let mut parts = vec![name.to_string()];
        for _ in 0..64 {
            if parent == 5 {
                return parts.into_iter().rev().collect::<Vec<_>>().join("/");
            }
            match recs.get(&parent).and_then(|r| r.name()) {
                Some((p, n)) if r_is_dir(&recs, parent) => {
                    parts.push(n);
                    parent = p;
                }
                _ => break,
            }
        }
        parts.push("$Orphan".into());
        parts.into_iter().rev().collect::<Vec<_>>().join("/")
    };
    let mut out = Vec::new();
    let mut keys: Vec<&u64> = recs.keys().collect();
    keys.sort();
    for i in keys {
        let r = &recs[i];
        if *i < 24 || r.in_use || r.is_dir {
            continue;
        }
        let list = record_attr_list(&recs, *i, src, &lcn_off, cs);
        let Some(data) = resolve_attr(&recs, &list, *i, 0x80, true) else { continue };
        let name = r
            .name()
            .or_else(|| list.iter().filter(|e| e.type_ == 0x30).find_map(|e| recs.get(&e.record).and_then(|x| x.name())));
        let path = match name {
            Some((parent, name)) => path_of(parent, &name),
            None => match index_names.get(i) {
                Some((parent, name)) => path_of(*parent, name),
                None => {
                    // Name removed on delete (Linux ntfs3): type from the contents.
                    let head = match &data {
                        Resolved::Resident(b) => b.iter().take(16).copied().collect(),
                        Resolved::Runs(runs, _) => match runs.first() {
                            Some((Some(l), _)) => read(src, lcn_off(*l), 16).unwrap_or_default(),
                            _ => Vec::new(),
                        },
                    };
                    format!("$NoName/record-{i}.{}", guess_ext(&head))
                }
            },
        };
        let (data, size, st) = match data {
            Resolved::Resident(bytes) => (Data::Resident(bytes.clone()), bytes.len() as u64, State::Intact),
            Resolved::Runs(runs, size) => {
                let mut ext = Vec::new();
                let (mut total, mut used) = (0usize, 0usize);
                let mut left = size;
                for (l, c) in &runs {
                    let bytes = (c * cs).min(left);
                    if bytes == 0 {
                        break;
                    }
                    match l {
                        Some(l) => {
                            for k in 0..bytes.div_ceil(cs) {
                                total += 1;
                                used += allocated(*l as u64 + k) as usize;
                            }
                            ext.push((Some(lcn_off(*l)), bytes));
                        }
                        None => ext.push((None, bytes)),
                    }
                    left -= bytes;
                }
                (Data::Extents(ext), size, state(used, total.max(1)))
            }
        };
        out.push(Deleted { volume: format!("{} (NTFS)", vol.label), path, size, state: st, data, note: None });
    }
    Ok((out, map))
}

/// File extension from the first bytes of a file.
pub fn guess_ext(head: &[u8]) -> &'static str {
    let starts = |m: &[u8]| head.starts_with(m);
    if starts(&[0xFF, 0xD8, 0xFF]) {
        "jpg"
    } else if starts(b"\x89PNG") {
        "png"
    } else if starts(b"%PDF") {
        "pdf"
    } else if starts(b"PK\x03\x04") {
        "zip"
    } else if starts(&[0xD0, 0xCF, 0x11, 0xE0]) {
        "doc"
    } else if starts(b"GIF8") {
        "gif"
    } else if starts(b"\x1f\x8b") {
        "gz"
    } else if head.len() >= 8 && &head[4..8] == b"ftyp" {
        "mp4"
    } else if !head.is_empty() && head.iter().all(|b| b.is_ascii_graphic() || b.is_ascii_whitespace()) {
        "txt"
    } else {
        "bin"
    }
}

fn r_is_dir(recs: &HashMap<u64, NtfsRecord>, i: u64) -> bool {
    recs.get(&i).is_some_and(|r| r.is_dir)
}

// ─── scan / recover ─────────────────────────────────────────────────────────

/// Result of scanning a source.
#[derive(Debug, Default)]
pub struct Scan {
    pub volumes: Vec<Volume>,
    pub files: Vec<Deleted>,
    /// Volumes whose filesystem has no undelete support (carving only).
    pub carve_only: Vec<String>,
    pub errors: Vec<String>,
    /// Allocation maps of the scanned volumes (for the block map).
    pub maps: Vec<VolMap>,
}

pub fn scan(src: &dyn Source) -> Scan {
    let mut s = Scan { volumes: volumes(src), ..Default::default() };
    for v in s.volumes.clone() {
        let r = match v.fs {
            "FAT32" => fat32(src, &v),
            "exFAT" => exfat(src, &v),
            "NTFS" => ntfs(src, &v),
            other => {
                s.carve_only.push(format!("{} ({other})", v.label));
                continue;
            }
        };
        match r {
            Ok((f, m)) => {
                s.files.extend(f);
                s.maps.push(m);
            }
            Err(e) => s.errors.push(format!("{} ({}): {e}", v.label, v.fs)),
        }
    }
    s
}

/// The bytes of a deleted file.
pub fn contents(src: &dyn Source, f: &Deleted) -> io::Result<Vec<u8>> {
    match &f.data {
        Data::Resident(b) => Ok(b.clone()),
        Data::Extents(ext) => {
            let mut out = Vec::with_capacity(f.size as usize);
            for (off, len) in ext {
                match off {
                    Some(o) => {
                        let n = (*len).min(src.len().saturating_sub(*o));
                        out.extend(read(src, *o, n as usize)?);
                    }
                    None => out.resize(out.len() + *len as usize, 0),
                }
            }
            out.truncate(f.size as usize);
            Ok(out)
        }
    }
}

/// Case-insensitive glob with `*` and `?`, matched against the whole path
/// and against the file name.
pub fn glob(pattern: &str, path: &str) -> bool {
    fn m(p: &[char], s: &[char]) -> bool {
        match (p.first(), s.first()) {
            (None, None) => true,
            (Some('*'), _) => m(&p[1..], s) || (!s.is_empty() && m(p, &s[1..])),
            (Some('?'), Some(_)) => m(&p[1..], &s[1..]),
            (Some(a), Some(b)) => a == b && m(&p[1..], &s[1..]),
            _ => false,
        }
    }
    let p: Vec<char> = pattern.to_lowercase().chars().collect();
    let full: Vec<char> = path.to_lowercase().chars().collect();
    let name: Vec<char> = path.rsplit('/').next().unwrap_or(path).to_lowercase().chars().collect();
    m(&p, &full) || m(&p, &name)
}

/// A path inside `dir` that does not exist yet (never overwrite).
fn free_path(dir: &std::path::Path, rel: &str) -> std::path::PathBuf {
    let clean: Vec<String> = rel
        .split('/')
        .filter(|c| !c.is_empty() && *c != "." && *c != "..")
        .map(|c| c.replace(['\0', '\\'], "_"))
        .collect();
    let mut p = dir.to_path_buf();
    for c in &clean {
        p.push(c);
    }
    if !p.exists() {
        return p;
    }
    let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let ext = p.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
    for n in 1.. {
        let q = p.with_file_name(format!("{stem} ({n}){ext}"));
        if !q.exists() {
            return q;
        }
    }
    unreachable!()
}

/// Write `files` into `dir` (folders kept). Returns (written path, bytes).
pub fn recover_to(src: &dyn Source, files: &[&Deleted], dir: &std::path::Path) -> Vec<Result<(String, u64), String>> {
    files
        .iter()
        .map(|f| {
            let rel = if f.volume.starts_with("whole device") {
                f.path.clone()
            } else {
                format!("{}/{}", f.volume.split(' ').take(2).collect::<Vec<_>>().join("-"), f.path)
            };
            let dest = free_path(dir, &rel);
            let bytes = contents(src, f).map_err(|e| format!("{}: {e}", f.path))?;
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
            }
            std::fs::write(&dest, &bytes).map_err(|e| format!("{}: {e}", dest.display()))?;
            Ok((dest.to_string_lossy().into_owned(), bytes.len() as u64))
        })
        .collect()
}

/// Demo scan (no disk): an NTFS volume with deleted files in every state.
pub fn demo_scan(size: u64) -> Scan {
    let vol = Volume { label: "partition 1".into(), fs: "NTFS", base: 1 << 20, size: size.saturating_sub(1 << 20) };
    let used = (0..MAP_CELLS)
        .map(|i| {
            let h = (i * 37 + 11) % 100;
            if i < 40 || (i < 700 && h < 70) { 1.0 } else if h < 12 { 0.5 } else { 0.0 }
        })
        .collect();
    let at = |cell: u64, len: u64| Data::Extents(vec![(Some(vol.base + vol.size / MAP_CELLS as u64 * cell), len)]);
    let file = |path: &str, size: u64, state: State, data: Data| Deleted {
        volume: format!("{} (NTFS)", vol.label),
        path: path.into(),
        size,
        state,
        data,
        note: None,
    };
    let span = vol.size / MAP_CELLS as u64;
    Scan {
        volumes: vec![vol.clone()],
        files: vec![
            file("Dokumen Kantor/Laporan Keuangan 2026.xlsx", 482_304, State::Intact, at(712, 482_304)),
            file("Dokumen Kantor/Kontrak Vendor.pdf", 2_310_144, State::Intact, at(750, span * 6)),
            file("Foto/IMG_2031.jpg", 3_145_728, State::PartlyReused, at(640, span * 9)),
            file("Foto/IMG_2032.jpg", 2_883_584, State::Overwritten, at(300, span * 4)),
            file("catatan.txt", 612, State::Intact, Data::Resident(b"demo".to_vec())),
            file("backup/db-2026-09-20.sql", 48_234_496, State::Intact, at(820, span * 30)),
        ],
        carve_only: Vec::new(),
        errors: Vec::new(),
        maps: vec![VolMap { volume: format!("{} (NTFS)", vol.label), base: vol.base, size: vol.size, used }],
    }
}

/// Map cells (of `cells`) a file's data occupies in its volume map.
pub fn file_cells(f: &Deleted, m: &VolMap, cells: usize) -> Vec<usize> {
    let Data::Extents(ext) = &f.data else { return Vec::new() };
    let mut v = Vec::new();
    for (off, len) in ext {
        let Some(o) = off else { continue };
        if *o < m.base || *o >= m.base + m.size {
            continue;
        }
        let a = ((*o - m.base) as u128 * cells as u128 / m.size.max(1) as u128) as usize;
        let b = (((*o - m.base + len.saturating_sub(1)) as u128 * cells as u128) / m.size.max(1) as u128) as usize;
        v.extend(a.min(cells - 1)..=b.min(cells - 1));
    }
    v.sort_unstable();
    v.dedup();
    v
}

// ─── carving ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Carved {
    pub offset: u64,
    pub len: u64,
    pub ext: &'static str,
}

const MAX_CARVE: u64 = 64 << 20;

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// End of a file starting at `head` (bytes from its start, up to MAX_CARVE).
fn carve_end(head: &[u8]) -> Option<(u64, &'static str)> {
    if head.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return find(head, &[0xFF, 0xD9]).map(|e| (e as u64 + 2, "jpg"));
    }
    if head.starts_with(b"\x89PNG\r\n\x1a\n") {
        return find(head, b"IEND").map(|e| (e as u64 + 8, "png"));
    }
    if head.starts_with(b"%PDF-") {
        let e = find(head, b"%%EOF")?;
        let mut end = e + 5;
        while end < head.len() && (head[end] == b'\r' || head[end] == b'\n') {
            end += 1;
        }
        return Some((end as u64, "pdf"));
    }
    if head.starts_with(b"PK\x03\x04") {
        let e = find(head, b"PK\x05\x06")?;
        let comment = head.get(e + 20..e + 22).map_or(0, |c| u16le(c, 0) as usize);
        let end = e + 22 + comment;
        let kind = if find(&head[..end.min(head.len())], b"word/").is_some() {
            "docx"
        } else if find(&head[..end.min(head.len())], b"xl/").is_some() {
            "xlsx"
        } else if find(&head[..end.min(head.len())], b"ppt/").is_some() {
            "pptx"
        } else {
            "zip"
        };
        return Some((end as u64, kind));
    }
    None
}

fn is_header(s: &[u8]) -> bool {
    s.starts_with(&[0xFF, 0xD8, 0xFF])
        || s.starts_with(b"\x89PNG\r\n\x1a\n")
        || s.starts_with(b"%PDF-")
        || s.starts_with(b"PK\x03\x04")
}

/// Group cluster indices `0..count` into byte ranges wherever `free` holds;
/// `off(c)` is the byte offset of cluster `c` (the end of a run is `off(c)` of
/// the first free cluster after it, so the ranges are contiguous).
fn cluster_runs(count: u64, free: impl Fn(u64) -> bool, off: impl Fn(u64) -> u64) -> Vec<(u64, u64)> {
    let mut v = Vec::new();
    let mut c = 0u64;
    while c < count {
        if free(c) {
            let start = c;
            while c < count && free(c) {
                c += 1;
            }
            v.push((off(start), off(c)));
        } else {
            c += 1;
        }
    }
    v
}

/// Free clusters of a volume as byte ranges, when its allocation table is
/// understood (NTFS `$Bitmap`, FAT32 table, exFAT bitmap). Unsupported
/// filesystems return an error and the caller carves the whole volume.
pub fn free_ranges(src: &dyn Source, vol: &Volume) -> io::Result<Vec<(u64, u64)>> {
    match vol.fs {
        "FAT32" => fat32_free(src, vol),
        "exFAT" => exfat_free(src, vol),
        "NTFS" => ntfs_free(src, vol),
        _ => Err(io::Error::other("no free-space bitmap for this filesystem")),
    }
}

fn fat32_free(src: &dyn Source, vol: &Volume) -> io::Result<Vec<(u64, u64)>> {
    let b = read(src, vol.base, 512)?;
    let bps = u16le(&b, 11) as u64;
    let spc = b[13] as u64;
    let reserved = u16le(&b, 14) as u64;
    let nfats = b[16] as u64;
    let fatsz = u32le(&b, 36) as u64;
    if bps == 0 || spc == 0 || fatsz == 0 {
        return Err(io::Error::other("bad FAT32 boot sector"));
    }
    let raw = read(src, vol.base + reserved * bps, (fatsz * bps) as usize)?;
    let fat: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let cs = bps * spc;
    let data = (reserved + nfats * fatsz) * bps;
    let count = (vol.size.saturating_sub(data) / cs).min(fat.len() as u64 - 2);
    let off = |c: u64| vol.base + data + c * cs;
    Ok(cluster_runs(count, |c| fat.get((c + 2) as usize).is_some_and(|v| v & 0x0FFF_FFFF == 0), off))
}

fn exfat_free(src: &dyn Source, vol: &Volume) -> io::Result<Vec<(u64, u64)>> {
    let b = read(src, vol.base, 512)?;
    let bps = 1u64 << b[108];
    let cs = bps << b[109];
    let fat_off = u32le(&b, 80) as u64 * bps;
    let heap = u32le(&b, 88) as u64 * bps;
    let count = u32le(&b, 92);
    let root = u32le(&b, 96);
    if cs == 0 || count == 0 {
        return Err(io::Error::other("bad exFAT boot sector"));
    }
    let fat_raw = read(src, vol.base + fat_off, (count as usize + 2) * 4)?;
    let fat: Vec<u32> = fat_raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let off = |c: u32| vol.base + heap + (c as u64 - 2) * cs;
    let valid = |c: u32| c >= 2 && c < count + 2;
    let chain = |mut c: u32| {
        let mut v = Vec::new();
        let mut seen = HashSet::new();
        while valid(c) && seen.insert(c) {
            v.push(c);
            c = fat[c as usize];
            if c >= 0xFFFF_FFF7 {
                break;
            }
        }
        v
    };
    let mut bitmap = Vec::new();
    let mut root_bytes = Vec::new();
    for cl in chain(root) {
        root_bytes.extend(read(src, off(cl), cs as usize).unwrap_or_default());
    }
    for e in root_bytes.chunks_exact(32) {
        if e[0] == 0x81 {
            let (c, len) = (u32le(e, 20), u64le(e, 24));
            let mut bm = Vec::new();
            for cl in chain(c) {
                bm.extend(read(src, off(cl), cs as usize).unwrap_or_default());
            }
            bm.truncate(len as usize);
            bitmap = bm;
            break;
        }
    }
    let in_use = |c: u64| bitmap.get((c / 8) as usize).is_some_and(|b| b & (1 << (c % 8)) != 0);
    let off2 = |c: u64| vol.base + heap + c * cs;
    Ok(cluster_runs(count as u64, |c| !in_use(c), off2))
}

fn ntfs_free(src: &dyn Source, vol: &Volume) -> io::Result<Vec<(u64, u64)>> {
    let b = read(src, vol.base, 512)?;
    let cs = (u16le(&b, 11) as u64) * b[13] as u64;
    let mft_lcn = u64le(&b, 48);
    let cpr = b[64] as i8;
    let rs = if cpr > 0 { cs * cpr as u64 } else { 1u64 << (-cpr as u32) };
    if cs == 0 || rs == 0 || rs > 65536 {
        return Err(io::Error::other("bad NTFS boot sector"));
    }
    let lcn_off = |l: i64| vol.base + l as u64 * cs;
    let mut rec0 = read(src, lcn_off(mft_lcn as i64), rs as usize)?;
    if !ntfs_fixup(&mut rec0) {
        return Err(io::Error::other("MFT record 0 unreadable"));
    }
    let rec0 = ntfs_record(&rec0);
    let mft_runs = rec0
        .attrs
        .iter()
        .find(|a| a.type_ == 0x80 && !a.named)
        .and_then(|a| a.runs.as_deref())
        .map(ntfs_runs)
        .ok_or_else(|| io::Error::other("no $MFT data runs"))?;
    // Record 6 holds $Bitmap; find it in the MFT by byte offset.
    let mut vcn = 0u64;
    let mut rec6 = None;
    for (l, c) in &mft_runs {
        let run = c * cs;
        if 6 * rs < vcn + run {
            if let Some(l) = l {
                let off = lcn_off(*l) + (6 * rs - vcn);
                if let Ok(mut r) = read(src, off, rs as usize) {
                    if ntfs_fixup(&mut r) {
                        rec6 = Some(ntfs_record(&r));
                    }
                }
            }
            break;
        }
        vcn += run;
    }
    let Some(rec6) = rec6 else {
        return Err(io::Error::other("no $Bitmap record"));
    };
    let attr6 = rec6.attrs.iter().find(|a| a.type_ == 0x80 && !a.named);
    let runs = attr6
        .and_then(|a| a.runs.as_deref())
        .map(ntfs_runs)
        .ok_or_else(|| io::Error::other("no $Bitmap data runs"))?;
    let bitmap_bytes = attr6.map(|a| a.size).filter(|s| *s > 0).unwrap_or_else(|| runs.iter().map(|(_, c)| c * cs).sum());
    let bitmap = read_runs(src, &runs, bitmap_bytes, &lcn_off, cs);
    let allocated = |c: u64| bitmap.get((c / 8) as usize).is_some_and(|b| b & (1 << (c % 8)) != 0);
    let count = vol.size / cs;
    let off = |c: u64| vol.base + c * cs;
    Ok(cluster_runs(count, |c| !allocated(c), off))
}

/// Search 512-byte-aligned file headers from `start` to `end` of the source.
pub fn carve(src: &dyn Source, start: u64, end: u64, progress: &mut dyn FnMut(u64, u64)) -> Vec<Carved> {
    let mut out = Vec::new();
    let chunk = 4u64 << 20;
    let mut off = start - start % 512;
    while off < end {
        let n = chunk.min(end - off) as usize;
        let Ok(buf) = read(src, off, n) else {
            off += chunk;
            continue;
        };
        let mut k = 0usize;
        while k + 512 <= buf.len() {
            if is_header(&buf[k..k + 8]) {
                let at = off + k as u64;
                let len = (MAX_CARVE).min(src.len() - at) as usize;
                if let Ok(head) = read(src, at, len) {
                    if let Some((l, ext)) = carve_end(&head) {
                        out.push(Carved { offset: at, len: l, ext });
                        // Skip past the file (whole sectors).
                        let skip = (l.div_ceil(512) * 512) as usize;
                        if k + skip <= buf.len() {
                            k += skip;
                            continue;
                        }
                        off = at + skip as u64 - chunk;
                        k = buf.len();
                        continue;
                    }
                }
            }
            k += 512;
        }
        off += chunk;
        progress(off.min(end) - start, end - start);
    }
    out
}

// ─── output directory safety ────────────────────────────────────────────────

/// Refuse a destination on the same physical disk as the source device.
#[cfg(unix)]
pub fn check_destination(source: &str, dest: &std::path::Path) -> Result<(), String> {
    use std::os::unix::fs::FileTypeExt;
    let meta = std::fs::metadata(source).map_err(|e| format!("{source}: {e}"))?;
    if !meta.file_type().is_block_device() {
        return Ok(()); // an image file: any destination is fine
    }
    let src_name = std::fs::canonicalize(source)
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_default();
    let src_disk = crate::recover::disk_name(&src_name);
    let dest_dev = crate::recover::mount_source(dest).ok_or("cannot tell which disk the destination is on")?;
    let dest_disk = crate::recover::disk_name(dest_dev.trim_start_matches("/dev/"));
    if dest_disk == src_disk {
        return Err(format!(
            "the destination {} is on /dev/{dest_disk}, the same disk as the source — writing there \
             can overwrite the very files you want back. Use another disk (USB drive, network share).",
            dest.display()
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
pub fn check_destination(_: &str, _: &std::path::Path) -> Result<(), String> {
    Ok(())
}

// ─── CLI ────────────────────────────────────────────────────────────────────

pub fn cmd(args: &[String]) -> i32 {
    let usage = "Usage: dcheck undelete <device | partition | image> [--to DIR] [--match PATTERN]\n\
                 \x20                     [--include-overwritten] [--carve [--free]]\n\n\
                 Lists deleted files (NTFS, FAT32, exFAT: names kept) and, with --to,\n\
                 recovers them into DIR, which must be on ANOTHER disk. The source is\n\
                 only read. Best: image the disk first (ddrescue) and work on the image.\n\n  \
                 --to DIR               recover into DIR (folders kept, nothing overwritten)\n  \
                 --match PATTERN        only files matching, e.g. '*.xlsx' or 'Dokumen*'\n  \
                 --include-overwritten  also write files whose space is in use again\n  \
                 --carve                ext4 / XFS / btrfs / unknown: search the raw data for\n                         \
                 JPEG, PNG, PDF and ZIP / Office files (names are lost); needs --to\n  \
                 --free                 with --carve: only search the free clusters of NTFS /\n                         \
                 FAT32 / exFAT (skip live files); other filesystems are carved whole";
    let (mut src, mut to, mut pat, mut all, mut carve_mode, mut free) = (None, None, None, false, false, false);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("{usage}");
                return 0;
            }
            "--to" => {
                i += 1;
                to = args.get(i).cloned();
            }
            "--match" => {
                i += 1;
                pat = args.get(i).cloned();
            }
            "--include-overwritten" => all = true,
            "--carve" => carve_mode = true,
            "--free" => free = true,
            f if f.starts_with('-') => {
                eprintln!("dcheck: unknown option '{f}'\n\n{usage}");
                return 2;
            }
            v => src = Some(v.to_string()),
        }
        i += 1;
    }
    let Some(src_path) = src else {
        eprintln!("{usage}");
        return 2;
    };
    #[cfg(unix)]
    let source = match FileSource::open(&src_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("dcheck: cannot open {src_path}: {e}");
            return 1;
        }
    };
    #[cfg(not(unix))]
    let source: Vec<u8> = {
        eprintln!("dcheck: undelete is not supported on this platform");
        return 1;
    };
    if let Some(dir) = &to {
        let d = std::path::Path::new(dir);
        if let Err(e) = std::fs::create_dir_all(d) {
            eprintln!("dcheck: cannot create {dir}: {e}");
            return 1;
        }
        if let Err(e) = check_destination(&src_path, d) {
            eprintln!("dcheck: refusing: {e}");
            return 1;
        }
    }
    if let Some(m) = crate::recover::mounted_rw(&src_path) {
        eprintln!(
            "warning: {src_path} is mounted read-write on {m}: the system keeps writing to it, which can \
             overwrite deleted files. Unmount it or image it first (ddrescue)."
        );
    }
    if carve_mode {
        let Some(dir) = &to else {
            eprintln!("dcheck: --carve needs --to DIR");
            return 2;
        };
        return carve_cmd(&source, std::path::Path::new(dir), free);
    }
    let s = scan(&source);
    for l in scan_lines(&s) {
        println!("{l}");
    }
    let Some(dir) = to else {
        if !s.files.is_empty() {
            println!("\n  Recover with: dcheck undelete {src_path} --to /path/on/another/disk [--match '*.docx']");
        }
        return 0;
    };
    let chosen: Vec<&Deleted> = s
        .files
        .iter()
        .filter(|f| pat.as_deref().is_none_or(|p| glob(p, &f.path)))
        .filter(|f| all || f.state != State::Overwritten)
        .collect();
    if chosen.is_empty() {
        println!("\n  Nothing to recover{}.", if pat.is_some() { " matching the pattern" } else { "" });
        return 1;
    }
    println!();
    let mut ok = 0;
    for r in recover_to(&source, &chosen, std::path::Path::new(&dir)) {
        match r {
            Ok((p, n)) => {
                ok += 1;
                println!("  recovered  {p}  ({})", human_size_bin(n));
            }
            Err(e) => println!("  failed     {e}"),
        }
    }
    println!("\n  {ok} of {} file(s) written to {dir}. Open them to check: a PARTLY REUSED file may be damaged.", chosen.len());
    if ok == chosen.len() {
        0
    } else {
        1
    }
}

/// The scan as a text report.
pub fn scan_lines(s: &Scan) -> Vec<String> {
    let mut out = vec![crate::report::section("DELETED FILES")];
    for v in &s.volumes {
        out.push(format!("  Volume       : {} — {}, {}", v.label, v.fs, human_size_bin(v.size)));
    }
    if s.volumes.is_empty() {
        out.push("  No filesystem found (no NTFS / FAT32 / exFAT / ext4 / XFS / btrfs signature).".into());
    }
    for e in &s.errors {
        out.push(format!("  Error        : {e}"));
    }
    for c in &s.carve_only {
        out.push(format!(
            "  {c}: deleted files keep no name or location on this filesystem — use --carve --to DIR"
        ));
    }
    if !s.files.is_empty() {
        out.push(String::new());
        out.push(format!("  {:<14} {:>10}  PATH", "STATE", "SIZE"));
        for f in &s.files {
            let note = f.note.map(|n| format!("  ({n})")).unwrap_or_default();
            out.push(format!("  {:<14} {:>10}  {}{note}", f.state.label(), human_size_bin(f.size), f.path));
        }
        let intact = s.files.iter().filter(|f| f.state == State::Intact).count();
        out.push(String::new());
        out.push(format!("  {} deleted file(s), {intact} intact.", s.files.len()));
    } else if !s.volumes.is_empty() && s.carve_only.len() < s.volumes.len() {
        out.push("  No deleted files found in the directory tables.".into());
    }
    out
}

fn carve_cmd(src: &dyn Source, dir: &std::path::Path, free: bool) -> i32 {
    let vols = volumes(src);
    let ranges: Vec<(u64, u64)> = if vols.is_empty() {
        vec![(0, src.len())]
    } else if free {
        let mut r = Vec::new();
        for v in &vols {
            match free_ranges(src, v) {
                Ok(rs) if !rs.is_empty() => r.extend(rs),
                Ok(_) => {}
                Err(e) => {
                    eprintln!("  {} ({}): {e} — carving the whole volume", v.label, v.fs);
                    r.push((v.base, v.base + v.size));
                }
            }
        }
        r
    } else {
        vols.iter().map(|v| (v.base, v.base + v.size)).collect()
    };
    let tty = std::io::IsTerminal::is_terminal(&std::io::stderr());
    let mut found = Vec::new();
    for (a, b) in ranges {
        let mut last = std::time::Instant::now();
        found.extend(carve(src, a, b, &mut |d, t| {
            if tty && last.elapsed().as_millis() > 500 {
                last = std::time::Instant::now();
                eprint!("\r  carving {} / {}", human_size_bin(d), human_size_bin(t));
            }
        }));
        if tty {
            eprint!("\r\x1b[2K");
        }
    }
    let mut n = 0;
    for (i, c) in found.iter().enumerate() {
        let dest = free_path(dir, &format!("carved/{:06}.{}", i + 1, c.ext));
        let data = read(src, c.offset, c.len as usize);
        match data.and_then(|d| {
            std::fs::create_dir_all(dest.parent().unwrap_or(dir))?;
            std::fs::write(&dest, d)
        }) {
            Ok(()) => {
                n += 1;
                println!("  carved  {}  ({}, at {})", dest.display(), human_size_bin(c.len), human_size_bin(c.offset));
            }
            Err(e) => println!("  failed  {}: {e}", dest.display()),
        }
    }
    println!("\n  {n} file(s) carved into {}. Names are lost; live files are found too.", dir.display());
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Load a `testdata/undelete/*.sparse` image (made on a real system:
    /// mkfs, write files, delete some, write another one).
    fn fixture(name: &str) -> Vec<u8> {
        let text = std::fs::read_to_string(format!("{}/testdata/undelete/{name}.sparse", env!("CARGO_MANIFEST_DIR"))).unwrap();
        let mut lines = text.lines();
        let size: usize = lines.next().unwrap().rsplit(' ').next().unwrap().parse().unwrap();
        let mut img = vec![0u8; size];
        for l in lines {
            let p: Vec<&str> = l.split(' ').collect();
            let s: usize = p[0].parse().unwrap();
            if p[1] == "H" {
                for (k, byte) in (0..p[2].len()).step_by(2).map(|i| u8::from_str_radix(&p[2][i..i + 2], 16).unwrap()).enumerate() {
                    img[s * 512 + k] = byte;
                }
            } else {
                let (count, b): (usize, u8) = (p[1].parse().unwrap(), u8::from_str_radix(p[3], 16).unwrap());
                img[s * 512..(s + count) * 512].fill(b);
            }
        }
        img
    }

    fn big() -> Vec<u8> {
        (0..20000u32).map(|i| ((i * 7 + 3) % 251) as u8).collect()
    }

    fn xlsx() -> Vec<u8> {
        let mut v = b"PK\x03\x04".to_vec();
        v.extend((0..6000u32).map(|i| ((i * 13 + 5) % 253) as u8));
        v
    }

    fn check(name: &str) -> Scan {
        let img = fixture(name);
        let s = scan(&img);
        assert!(s.errors.is_empty(), "{:?}", s.errors);
        let by = |p: &str| s.files.iter().find(|f| f.path == p).unwrap_or_else(|| panic!("{name}: {p} not found in {:#?}", s.files.iter().map(|f| (&f.path, f.state)).collect::<Vec<_>>()));
        let b = by("big.bin");
        assert_eq!(b.size, 20000);
        if b.state == State::Intact {
            assert_eq!(contents(&img, b).unwrap(), big(), "{name}: big.bin");
        }
        let x = by("Dokumen Kantor/Laporan Keuangan 2026.xlsx");
        assert_eq!(x.size, 6004);
        if x.state == State::Intact {
            assert_eq!(contents(&img, x).unwrap(), xlsx(), "{name}: xlsx");
        }
        // Live files are never listed.
        assert!(s.files.iter().all(|f| !f.path.ends_with("keep.txt") && !f.path.ends_with("after.txt")));
        s
    }

    #[test]
    fn fat32_lists_and_recovers_deleted_files_with_long_names() {
        let s = check("fat32");
        assert_eq!(s.volumes[0].fs, "FAT32");
        let img = fixture("fat32");
        let n = s.files.iter().find(|f| f.path == "note.txt").unwrap();
        assert_eq!(contents(&img, n).unwrap(), b"short note, deleted\n");
        assert!(s.files.iter().all(|f| f.state == State::Intact));
    }

    #[test]
    fn exfat_lists_and_recovers_deleted_files() {
        let s = check("exfat");
        assert_eq!(s.volumes[0].fs, "exFAT");
        // note.txt's cluster went to after.txt (written after the delete).
        let n = s.files.iter().find(|f| f.path == "note.txt").unwrap();
        assert_eq!(n.state, State::Overwritten);
        assert_eq!(s.maps.len(), 1);
        assert!(s.maps[0].used.iter().any(|u| *u > 0.0));
    }

    #[test]
    fn ntfs_lists_and_recovers_deleted_files_with_folders() {
        let s = check("ntfs");
        assert_eq!(s.volumes[0].fs, "NTFS");
        // Small files are resident in the MFT record.
        let img = fixture("ntfs");
        let h = s.files.iter().find(|f| f.path == "hello.txt").unwrap();
        assert!(matches!(h.data, Data::Resident(_)));
        assert_eq!(contents(&img, h).unwrap(), b"hello from dcheck\n".repeat(3));
        assert!(s.files.iter().all(|f| f.state == State::Intact));
    }

    #[test]
    fn ntfs3_deleted_files_without_names_are_still_recovered() {
        // The Linux ntfs3 driver drops $FILE_NAME on delete; $DATA stays.
        let img = fixture("ntfs3");
        let s = scan(&img);
        assert!(s.errors.is_empty(), "{:?}", s.errors);
        let names: Vec<&str> = s.files.iter().map(|f| f.path.as_str()).collect();
        assert!(names.iter().all(|n| n.starts_with("$NoName/record-")), "{names:?}");
        let b = s.files.iter().find(|f| f.size == 20000).expect("big.bin by size");
        assert!(b.path.ends_with(".bin"), "{}", b.path);
        if b.state == State::Intact {
            assert_eq!(contents(&img, b).unwrap(), big());
        }
        let x = s.files.iter().find(|f| f.size == 6004).expect("xlsx by size");
        assert!(x.path.ends_with(".zip"), "{}", x.path);
        assert!(s.files.iter().any(|f| f.path.ends_with(".txt")), "{names:?}");
    }

    #[test]
    #[ignore]
    fn print_fixture_scans() {
        for n in ["fat32", "exfat", "ntfs", "ntfs3"] {
            for l in scan_lines(&scan(&fixture(n))) {
                eprintln!("{n}: {l}");
            }
        }
    }

    #[test]
    fn ntfs_runlist_and_fixup() {
        // 0x21: 1-byte length, 2-byte offset: 0x18 clusters at LCN 0x5634;
        // then 0x11: 1/1 with a negative offset (-2).
        let runs = ntfs_runs(&[0x21, 0x18, 0x34, 0x56, 0x11, 0x04, 0xFE, 0x01, 0x03, 0x00]);
        assert_eq!(runs, vec![(Some(0x5634), 0x18), (Some(0x5632), 4), (None, 3)]);
        let mut rec = vec![0u8; 1024];
        rec[..4].copy_from_slice(b"FILE");
        rec[4] = 48;
        rec[6] = 3;
        rec[48..50].copy_from_slice(&[0xAB, 0xCD]);
        rec[50..52].copy_from_slice(&[1, 2]);
        rec[52..54].copy_from_slice(&[3, 4]);
        rec[510..512].copy_from_slice(&[0xAB, 0xCD]);
        rec[1022..1024].copy_from_slice(&[0xAB, 0xCD]);
        assert!(ntfs_fixup(&mut rec));
        assert_eq!(&rec[510..512], &[1, 2]);
        assert_eq!(&rec[1022..1024], &[3, 4]);
    }

    #[test]
    fn finds_partitions_in_a_whole_disk_image() {
        // MBR with the FAT32 fixture as partition 1 at LBA 2048.
        let part = fixture("fat32");
        let mut disk = vec![0u8; 2048 * 512];
        disk[446 + 4] = 0x0C;
        disk[446 + 8..446 + 12].copy_from_slice(&2048u32.to_le_bytes());
        disk[446 + 12..446 + 16].copy_from_slice(&((part.len() / 512) as u32).to_le_bytes());
        disk[510] = 0x55;
        disk[511] = 0xAA;
        disk.extend(&part);
        let s = scan(&disk);
        assert_eq!(s.volumes.len(), 1);
        assert_eq!(s.volumes[0].label, "partition 1");
        let b = s.files.iter().find(|f| f.path == "big.bin").unwrap();
        if b.state == State::Intact {
            assert_eq!(contents(&disk, b).unwrap(), big());
        }
    }

    #[test]
    fn carves_known_formats() {
        let mut img = vec![0u8; 8192];
        let jpg = [0xFF, 0xD8, 0xFF, 0xE0, 1, 2, 3, 0xFF, 0xD9];
        img[512..512 + jpg.len()].copy_from_slice(&jpg);
        let pdf = b"%PDF-1.4\nhello\n%%EOF\n";
        img[2048..2048 + pdf.len()].copy_from_slice(pdf);
        let found = carve(&img, 0, img.len() as u64, &mut |_, _| {});
        assert_eq!(found, vec![Carved { offset: 512, len: 9, ext: "jpg" }, Carved { offset: 2048, len: pdf.len() as u64, ext: "pdf" }]);
    }

    #[test]
    fn globs_and_safe_paths() {
        assert!(glob("*.xlsx", "Dokumen Kantor/Laporan Keuangan 2026.xlsx"));
        assert!(glob("laporan*", "Dokumen Kantor/Laporan Keuangan 2026.xlsx"));
        assert!(glob("dokumen*/*", "Dokumen Kantor/x.txt"));
        assert!(!glob("*.pdf", "a.xlsx"));
        let dir = std::env::temp_dir().join(format!("dcheck-undelete-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = free_path(&dir, "../../etc/passwd");
        assert!(p.starts_with(&dir));
        std::fs::write(dir.join("a.txt"), "x").unwrap();
        assert_eq!(free_path(&dir, "a.txt").file_name().unwrap(), "a (1).txt");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // ── NTFS $ATTRIBUTE_LIST / index slack ──────────────────────────────────

    fn attr_resident(type_: u32, id: u16, named: bool, value: &[u8]) -> Vec<u8> {
        let (co, len) = (24usize, 24 + value.len());
        let mut a = vec![0u8; len];
        a[0..4].copy_from_slice(&type_.to_le_bytes());
        a[4..8].copy_from_slice(&(len as u32).to_le_bytes());
        a[8] = 0;
        a[9] = named as u8;
        a[14..16].copy_from_slice(&id.to_le_bytes());
        a[16..20].copy_from_slice(&(value.len() as u32).to_le_bytes());
        a[20..22].copy_from_slice(&(co as u16).to_le_bytes());
        a[24..].copy_from_slice(value);
        a
    }

    fn attr_nonresident(type_: u32, id: u16, vcn: u64, size: u64, runs: &[u8]) -> Vec<u8> {
        let (ro, len) = (64usize, 64 + runs.len());
        let mut a = vec![0u8; len];
        a[0..4].copy_from_slice(&type_.to_le_bytes());
        a[4..8].copy_from_slice(&(len as u32).to_le_bytes());
        a[8] = 1;
        a[14..16].copy_from_slice(&id.to_le_bytes());
        a[16..24].copy_from_slice(&vcn.to_le_bytes());
        a[24..32].copy_from_slice(&vcn.to_le_bytes());
        a[32..34].copy_from_slice(&(ro as u16).to_le_bytes());
        a[48..56].copy_from_slice(&size.to_le_bytes());
        a[ro..].copy_from_slice(runs);
        a
    }

    fn record(flags: u16, attrs: &[Vec<u8>]) -> Vec<u8> {
        let mut rec = vec![0u8; 1024];
        rec[..4].copy_from_slice(b"FILE");
        rec[20..22].copy_from_slice(&48u16.to_le_bytes());
        rec[22..24].copy_from_slice(&flags.to_le_bytes());
        let mut off = 48usize;
        for a in attrs {
            rec[off..off + a.len()].copy_from_slice(a);
            off += a.len();
        }
        rec[off..off + 4].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        rec
    }

    fn list_entry(type_: u32, id: u16, record: u64, vcn: u64) -> Vec<u8> {
        let mut e = vec![0u8; 26];
        e[0..4].copy_from_slice(&type_.to_le_bytes());
        e[4..6].copy_from_slice(&26u16.to_le_bytes());
        e[8..16].copy_from_slice(&vcn.to_le_bytes());
        e[16..24].copy_from_slice(&record.to_le_bytes());
        e[24..26].copy_from_slice(&id.to_le_bytes());
        e
    }

    #[test]
    fn follows_attribute_list_across_extension_records() {
        // A highly fragmented $DATA split into two extents, the second one in
        // another MFT record. Each extent runlist is absolute on its own.
        let list: Vec<u8> = [list_entry(0x80, 1, 41, 0), list_entry(0x80, 2, 42, 2)].concat();
        let base = record(0, &[attr_resident(0x20, 0, false, &list)]);
        // LCN 100, 2 clusters (len 1 byte, offset 1 byte).
        let ext1 = record(0, &[attr_nonresident(0x80, 1, 0, 3 * 4096, &[0x11, 2, 100, 0x00])]);
        // LCN 500, 1 cluster: absolute, not relative to the previous extent.
        let ext2 = record(0, &[attr_nonresident(0x80, 2, 2, 0, &[0x21, 1, 0xF4, 0x01, 0x00])]);
        let mut recs = HashMap::new();
        recs.insert(40u64, ntfs_record(&base));
        recs.insert(41u64, ntfs_record(&ext1));
        recs.insert(42u64, ntfs_record(&ext2));
        let parsed = attr_list_entries(&recs[&40].attrs[0].resident.clone().unwrap());
        assert_eq!(parsed, vec![
            AttrRef { type_: 0x80, id: 1, named: false, record: 41, vcn: 0 },
            AttrRef { type_: 0x80, id: 2, named: false, record: 42, vcn: 2 },
        ]);
        match resolve_attr(&recs, &parsed, 40, 0x80, true) {
            Some(Resolved::Runs(runs, size)) => {
                assert_eq!(runs, vec![(Some(100), 2), (Some(500), 1)]);
                assert_eq!(size, 3 * 4096);
            }
            other => panic!("expected merged runs, got {other:?}"),
        }
    }

    fn file_name_value(parent: u64, name: &str) -> Vec<u8> {
        let units: Vec<u16> = name.encode_utf16().collect();
        let mut v = vec![0u8; 66 + units.len() * 2];
        v[0..8].copy_from_slice(&parent.to_le_bytes());
        v[64] = units.len() as u8;
        v[65] = 1;
        for (k, u) in units.iter().enumerate() {
            v[66 + 2 * k..68 + 2 * k].copy_from_slice(&u.to_le_bytes());
        }
        v
    }

    fn index_entry(rec: u64, parent: u64, name: &str) -> Vec<u8> {
        let fname = file_name_value(parent, name);
        let mut e = vec![0u8; 16 + fname.len()];
        e[0..8].copy_from_slice(&rec.to_le_bytes());
        e[8..10].copy_from_slice(&((16 + fname.len()) as u16).to_le_bytes());
        e[10..12].copy_from_slice(&(fname.len() as u16).to_le_bytes());
        e[16..].copy_from_slice(&fname);
        e
    }

    #[test]
    fn reads_names_from_index_root_and_allocation() {
        let mut root = vec![0u8; 0x20];
        root[0..4].copy_from_slice(&0x30u32.to_le_bytes());
        root[8..12].copy_from_slice(&512u32.to_le_bytes());
        root[0x10..0x14].copy_from_slice(&16u32.to_le_bytes()); // entries at 0x20
        root.extend(index_entry(77, 5, "laporan.xlsx"));
        root.extend([0u8; 12]);
        root.extend(2u32.to_le_bytes()); // last entry flag
        assert_eq!(index_root_entries(&root), vec![(77, 5, "laporan.xlsx".into())]);

        // An INDX block with the update-sequence fixup applied, entries at 0x38.
        let mut block = vec![0u8; 512];
        block[..4].copy_from_slice(b"INDX");
        block[4..6].copy_from_slice(&48u16.to_le_bytes());
        block[6..8].copy_from_slice(&2u16.to_le_bytes());
        block[48..50].copy_from_slice(&[0xAB, 0xCD]);
        block[0x18..0x1C].copy_from_slice(&32u32.to_le_bytes()); // entries at 0x38
        let e = index_entry(78, 5, "photo.jpg");
        block[0x38..0x38 + e.len()].copy_from_slice(&e);
        let last = 0x38 + e.len();
        block[last + 12..last + 16].copy_from_slice(&2u32.to_le_bytes());
        block[510..512].copy_from_slice(&[0xAB, 0xCD]);
        assert_eq!(index_alloc_entries(&block, 512), vec![(78, 5, "photo.jpg".into())]);
    }

    #[test]
    fn free_ranges_exclude_allocated_clusters() {
        for n in ["fat32", "exfat", "ntfs"] {
            let img = fixture(n);
            let vols = volumes(&img);
            assert_eq!(vols.len(), 1, "{n}");
            let ranges = free_ranges(&img, &vols[0]).unwrap_or_else(|e| panic!("{n}: {e}"));
            assert!(!ranges.is_empty(), "{n}: no free ranges");
            let mut prev = vols[0].base;
            for (a, b) in &ranges {
                assert!(a >= &prev && b > a, "{n}: bad range {a}..{b}");
                assert!(b <= &(vols[0].base + vols[0].size), "{n}: range past volume");
                prev = *b;
            }
            assert!(ranges.len() > 1, "{n}: allocation should have gaps");
        }
    }

    #[test]
    fn free_carving_skips_allocated_space() {
        let mut img = fixture("fat32");
        let vols = volumes(&img);
        let ranges = free_ranges(&img, &vols[0]).unwrap();
        let jpg: Vec<u8> = [&[0xFF, 0xD8, 0xFF][..], &[7u8; 60][..], &[0xFF, 0xD9][..]].concat();
        // One header in free space, one in the allocated gap before it.
        let (free_at, _) = ranges[0];
        // The first byte after a free run starts an allocated cluster.
        let gap = ranges[0].1;
        assert_ne!(gap, free_at);
        for at in [free_at as usize, gap as usize] {
            if at + jpg.len() <= img.len() {
                img[at..at + jpg.len()].copy_from_slice(&jpg);
            }
        }
        let mut found = Vec::new();
        for (a, b) in &ranges {
            found.extend(carve(&img, *a, *b, &mut |_, _| {}));
        }
        assert!(found.iter().any(|c| c.offset == free_at), "free jpg not found: {found:?}");
        assert!(!found.iter().any(|c| c.offset == gap as u64), "allocated jpg was carved");
        assert!(gap >= vols[0].base);
    }

    #[test]
    fn contiguity_assumption_is_noted() {
        // big.bin needs several clusters: FAT32 has no per-file chain left, so
        // it says the location is an assumption.
        let s = scan(&fixture("fat32"));
        let b = s.files.iter().find(|f| f.path == "big.bin").unwrap();
        assert_eq!(b.note, Some("assumed contiguous"));
        // exFAT keeps NoFatChain: a contiguous deleted file needs no assumption.
        let s = scan(&fixture("exfat"));
        let b = s.files.iter().find(|f| f.size == 20000).unwrap();
        assert_eq!(b.note, None);
    }
}
