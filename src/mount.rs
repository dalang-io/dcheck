//! Filesystem usage for mounted partitions (`statvfs` on Linux, `df` elsewhere).

#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

#[cfg(all(target_os = "linux", not(target_pointer_width = "64")))]
compile_error!("dcheck's raw ioctl/statvfs structs assume a 64-bit Linux target");

#[derive(Debug, Clone, Copy)]
pub struct Usage {
    pub total: u64,
    pub used: u64,
    pub avail: u64,
    pub percent: f64,
}

/// Compute usage from raw `statvfs` counters (used% as `df` does).
pub fn from_blocks(frsize: u64, blocks: u64, bfree: u64, bavail: u64) -> Usage {
    let frsize = frsize.max(1);
    let total = blocks.saturating_mul(frsize);
    let free = bfree.saturating_mul(frsize);
    let avail = bavail.saturating_mul(frsize);
    let used = total.saturating_sub(free);
    let denom = used.saturating_add(avail);
    let percent = if denom > 0 {
        used as f64 * 100.0 / denom as f64
    } else {
        0.0
    };
    Usage {
        total,
        used,
        avail,
        percent,
    }
}

#[cfg(target_os = "linux")]
pub fn usage(mount: &str) -> Option<Usage> {
    use std::ffi::CString;

    #[repr(C)]
    struct Statvfs {
        f_bsize: u64,
        f_frsize: u64,
        f_blocks: u64,
        f_bfree: u64,
        f_bavail: u64,
        f_files: u64,
        f_ffree: u64,
        f_favail: u64,
        f_fsid: u64,
        f_flag: u64,
        f_namemax: u64,
        __f_spare: [u32; 6],
    }

    extern "C" {
        fn statvfs(path: *const std::ffi::c_char, buf: *mut Statvfs) -> i32;
    }

    let c = CString::new(mount).ok()?;
    let mut st = Statvfs {
        f_bsize: 0,
        f_frsize: 0,
        f_blocks: 0,
        f_bfree: 0,
        f_bavail: 0,
        f_files: 0,
        f_ffree: 0,
        f_favail: 0,
        f_fsid: 0,
        f_flag: 0,
        f_namemax: 0,
        __f_spare: [0; 6],
    };
    let rc = unsafe { statvfs(c.as_ptr(), &mut st) };
    if rc != 0 {
        return None;
    }
    Some(from_blocks(st.f_frsize, st.f_blocks, st.f_bfree, st.f_bavail))
}

#[cfg(not(target_os = "linux"))]
pub fn usage(mount: &str) -> Option<Usage> {
    // macOS and the BSDs have no portably-declared statvfs struct here; `df -P
    // -k` reports the same counters in 1024-byte blocks.
    let out = std::process::Command::new("df").args(["-P", "-k", mount]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    parse_df(&String::from_utf8_lossy(&out.stdout))
}

/// Parse the data line of `df -P -k` (frsize is 1024 bytes).
#[cfg(not(target_os = "linux"))]
pub fn parse_df(text: &str) -> Option<Usage> {
    let line = text.lines().skip(1).find(|l| !l.trim().is_empty())?;
    let f: Vec<&str> = line.split_whitespace().collect();
    if f.len() < 4 {
        return None;
    }
    let blocks: u64 = f[1].parse().ok()?;
    let used: u64 = f[2].parse().ok()?;
    let avail: u64 = f[3].parse().ok()?;
    Some(from_blocks(1024, blocks, blocks.saturating_sub(used), avail))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_percent_like_df() {
        // 100 blocks, 40 free (10 reserved), 4096 bytes: used=60, avail=30.
        let u = from_blocks(4096, 100, 40, 30);
        assert_eq!(u.total, 100 * 4096);
        assert_eq!(u.used, 60 * 4096);
        assert_eq!(u.avail, 30 * 4096);
        assert!((u.percent - 66.67).abs() < 0.5); // 60 / (60+30)
    }

    #[test]
    #[cfg(not(target_os = "linux"))]
    fn parses_df_output() {
        let out = "Filesystem 1024-blocks      Used Available Capacity Mounted on\n\
                   /dev/disk3s1   1000000    600000    300000      67%   /\n";
        let u = parse_df(out).unwrap();
        assert_eq!(u.total, 1_000_000 * 1024);
        assert_eq!(u.used, 600_000 * 1024);
        assert_eq!(u.avail, 300_000 * 1024);
        assert!((u.percent - 66.67).abs() < 0.5);
        assert!(parse_df("garbage").is_none());
    }
}
