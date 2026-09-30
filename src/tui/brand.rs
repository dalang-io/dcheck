//! dcheck's own [`wayang_tui`] brand bits.
//!
//! The palette, glyph set, borders, corners, gauge/field/caption/badge/keycap
//! rendering and the panel frame geometry all come from the shared crate —
//! dcheck renders through the *same* code as the wayang HUDs, so the canonical
//! visual spec (`wayangos/docs/TUI-UX-REVAMP.md` §5) has to change in one place
//! instead of drifting in two.
//!
//! What stays here is the three things that are genuinely dcheck's:
//!
//! 1. the [`DCHECK`] identity — the colour override is `DCHECK_COLOR`, not
//!    `WAYANG_FW_COLOR`, and the brand word is `DCHECK`;
//! 2. the logotype, which the crate's pixel-font `logo()` cannot draw (see
//!    [`logo`]);
//! 3. the panel-title style, which the crate's focus-model `panel()` would
//!    change (see [`panel`]).
//!
//! Everything else is re-exported from [`wayang_tui::widgets`] so `views.rs`
//! keeps its short `w::` call sites.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders};
use wayang_tui::theme::{App, Theme};

/// dcheck's identity: `DCHECK_COLOR=truecolor|ansi|none`, tool `dcheck`, brand
/// word `DCHECK` (used by the header bar and the logotype).
pub const DCHECK: App = App {
    env_prefix: "DCHECK",
    tool: "dcheck",
    brand: "DCHECK",
};

// ─── logotype ───────────────────────────────────────────────────────────────

/// The `DCHECK` logotype: two rows of half-blocks (fancy) or one ASCII line
/// (`--plain`).
///
/// **Kept local, deliberately.** `wayang_tui::widgets::logo()` rasterises a word
/// through a hand-drawn 4-row bitmap table that covers the *wayang* brands only
/// — `D`, `C`, `H` and `K` are not in it, so `logo("DCHECK", …)` renders
/// `??????` until the table learns four new glyphs. dcheck's own mark is a fixed
/// pair of rows; [`logo_lines`] then applies the crate's colouring rule.
pub fn logo(theme: &Theme) -> &'static [&'static str] {
    if theme.is_plain() {
        &["== D C H E C K =="]
    } else {
        &["█▀▄ █▀▀ █ █ █▀▀ █▀▀ █▄▀", "█▄▀ █▄▄ █▀█ ██▄ █▄▄ █ █"]
    }
}

/// [`logo`] as styled lines, using the crate's rule: row 0 `accent`, every row
/// below it `accent2`, both bold (including a one-row `--plain` logotype).
pub fn logo_lines(theme: &Theme) -> Vec<Line<'static>> {
    logo(theme)
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let p = &theme.palette;
            let color = if i == 0 { p.accent } else { p.accent2 };
            Line::from(Span::styled(*row, p.bold(color)))
        })
        .collect()
}

// ─── panel ──────────────────────────────────────────────────────────────────

/// Draw dcheck's HUD panel (thin frame, heavy accent corners, `◢ TITLE ◣`) and
/// return its inner area. `right` is an optional right-aligned title.
///
/// **Kept local, deliberately.** The crate's `panel`/`panel_focused` implement
/// the *focused-pane* model of the spec: an unfocused pane gets a `dim` title,
/// a focused one an `accent` title **plus** a `▸ ` prefix and an `accent`
/// frame. dcheck has no focus ring — its one keyboard-owning pane is the device
/// table, everything else is a read-only detail card — so there is nothing to
/// mark and every title is `accent`+bold. Calling the crate's `panel(.., false,
/// ..)` would grey out every title; calling it with `true` would add a `▸ `
/// marker to all of them and recolour every frame.
///
/// The geometry is still the shared one: `theme.palette.border` frame,
/// `theme.ui.border()` set, `theme.ui.corners()` brackets and
/// `theme.palette.base()` fill. If dcheck ever gains a focus ring, delete this
/// and call the crate's `panel` / `panel_focused`.
pub fn panel(
    f: &mut Frame,
    area: Rect,
    title: &str,
    right: Option<Line<'static>>,
    theme: &Theme,
) -> Rect {
    let p = &theme.palette;
    let ui = theme.ui;
    let title_line = if ui.plain {
        Line::from(Span::styled(format!("[ {title} ]"), p.bold(p.accent)))
    } else {
        Line::from(vec![
            Span::styled("◢ ", p.fg(p.accent2)),
            Span::styled(title.to_string(), p.bold(p.accent)),
            Span::styled(" ◣", p.fg(p.accent2)),
        ])
    };
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_set(ui.border())
        .border_style(p.fg(p.border))
        .style(p.base())
        .title(title_line);
    if let Some(r) = right {
        block = block.title_top(r.right_aligned());
    }
    let inner = block.inner(area);
    f.render_widget(block, area);

    if let Some([tl, tr, bl, br]) = ui.corners() {
        if area.width >= 2 && area.height >= 2 {
            let (x0, y0) = (area.x, area.y);
            let (x1, y1) = (area.right() - 1, area.bottom() - 1);
            let style = p.fg(p.accent);
            let buf = f.buffer_mut();
            for (x, y, sym) in [(x0, y0, tl), (x1, y0, tr), (x0, y1, bl), (x1, y1, br)] {
                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.set_symbol(sym).set_style(style);
                }
            }
        }
    }
    inner
}

#[cfg(test)]
mod tests {
    use super::*;
    use wayang_tui::theme::{ColorMode, Flags};

    /// The generic colour resolution (`NO_COLOR` > `DCHECK_COLOR` >
    /// `COLORTERM`) is the shared crate's and is tested there. What is dcheck's
    /// own, and untested there, is the *brand identity* the crate reads its
    /// override from — if `DCHECK.env_prefix` were ever wrong, `DCHECK_COLOR`
    /// would silently stop working and no generic test would notice.
    #[test]
    fn the_brand_carries_dchecks_own_env_prefix() {
        assert_eq!(DCHECK.env_prefix, "DCHECK");
        assert_eq!(DCHECK.tool, "dcheck");
        assert_eq!(DCHECK.brand, "DCHECK");
    }

    /// `DCHECK_COLOR` must reach the palette through the brand, and the
    /// fallbacks must stay in dcheck's order.
    #[test]
    fn dcheck_color_drives_the_palette_and_falls_back() {
        let mk = |no_color: bool, own: Option<&str>, term: Option<&str>| {
            Theme::from_env(DCHECK, Flags::default(), no_color, own, term)
                .palette
                .mode
        };
        // NO_COLOR wins over everything.
        assert_eq!(
            mk(true, Some("truecolor"), Some("truecolor")),
            ColorMode::Mono
        );
        // then DCHECK_COLOR, then COLORTERM, then a plain ANSI default.
        assert_eq!(mk(false, Some("ansi"), Some("truecolor")), ColorMode::Ansi);
        assert_eq!(mk(false, Some("truecolor"), None), ColorMode::Neon);
        assert_eq!(mk(false, None, Some("24bit")), ColorMode::Neon);
        assert_eq!(mk(false, None, None), ColorMode::Ansi);
    }
}
