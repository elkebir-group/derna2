//! Settings read from the environment for development: phase timers,
//! diagnostic printouts, alternative code paths kept for A/B comparison,
//! cross-checks, and knobs that tune the search's cost or stop it early.
//!
//! Every such read goes through here, so one switch decides whether the
//! environment is consulted at all. A build without development settings
//! answers each read as if the setting were unset, and the library then
//! behaves exactly as it does with none of them set. Each read names a
//! [`Setting`] rather than a string, and only a build that reads the
//! environment knows which variable a setting is.

use std::env::VarError;
use std::ffi::OsString;

/// One development setting. A read of it answers as unset unless the build
/// reads development settings; the code at each read says what the unset
/// answer is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Setting {
    ConstrainedLz,
    ConstrainedSeparate,
    FanDetail,
    FanWorkers,
    FillFanAll,
    FillSerialIndex,
    FoldLog,
    FoldLogMax,
    FoldLogSkip,
    GridLanes,
    MaoTimes,
    MotifAwareHeadroom,
    MotifAwareMargin,
    MotifBracketCuts,
    MotifCellFoldDe,
    MotifCellFoldTie,
    MotifDensityCarry,
    MotifFrontCensus,
    MotifGapBudget,
    MotifGapCarry,
    MotifGapCarryAdapt,
    MotifGapDensity,
    MotifGapDouble,
    MotifGapFit,
    MotifGapFloorHeld,
    MotifGapFolds,
    MotifGapMidslice,
    MotifGapMidsliceSecs,
    MotifGapOrder,
    MotifGapReuse,
    MotifGapRoom,
    MotifGapSecs,
    MotifPassCache,
    MotifPassGrid,
    MotifPassLanes,
    MotifPassResident,
    MotifProbeLearn,
    MotifProbeSkip,
    MotifSliceGrow,
    MotifSliceShrink,
    MotifSliceStep,
    MotifStopOpen,
    OnesideValidate,
    OutsidePhases,
    Profile,
    PullFill,
    PushPool,
    PushScan,
    SerialDetail,
    SpanCount,
    SpanEager,
    SplitCensus,
    SplitPrune,
    SplitValidate,
    SweepSkip,
    WalkFloorCut,
    WalkMark,
    WalkPassGrid,
    WalkStopSlack,
    WitnessLog,
    WitnessLogMax,
    WitnessLogSkip,
}

/// Unset: this build reads no development setting.
pub fn var(_setting: Setting) -> Result<String, VarError> {
    Err(VarError::NotPresent)
}

/// Unset: this build reads no development setting.
pub fn var_os(_setting: Setting) -> Option<OsString> {
    None
}
