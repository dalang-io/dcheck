//! dcheck — device health check, as a library.
//!
//! The crate root exposes the engine; `src/main.rs` is a thin CLI on top of it:
//! argument parsing, exit codes and the SIGPIPE fixup are process concerns, not
//! something a library caller wants. Keeping the split means dcheck can be
//! depended on (tests, the landing-page screenshot job, another HUD) without
//! dragging a `main` along, and it is the same shape the wayang products use.
//!
//! The visual layer ([`tui`]) renders through the shared
//! [`wayang_tui`] component library — dcheck's palette, glyphs and widgets are
//! the canonical ones from `wayangos/docs/TUI-UX-REVAMP.md` §5, not a copy.

#![allow(clippy::collapsible_if)]

pub mod authenticity;
pub mod bench;
pub mod board;
pub mod cache;
pub mod config;
pub mod cpu;
pub mod enumerate;
pub mod health;
pub mod ipmi;
pub mod json;
pub mod kernlog;
pub mod model;
pub mod monitor;
pub mod mount;
pub mod native;
pub mod oui_table;
pub mod ram;
pub mod recover;
pub mod report;
pub mod smartctl;
pub mod tui;
pub mod undelete;
pub mod update;
pub mod verify;
pub mod virt;
