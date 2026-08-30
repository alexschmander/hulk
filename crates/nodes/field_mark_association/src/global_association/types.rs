pub(crate) const FEATURE_CLASSES: [VisualFeatureClass; 5] = [
    VisualFeatureClass::GoalPost,
    VisualFeatureClass::LSpot,
    VisualFeatureClass::TSpot,
    VisualFeatureClass::XSpot,
    VisualFeatureClass::PenaltySpot,
];

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
/// Field-feature classes supported by the global association solver.
pub enum VisualFeatureClass {
    /// Upright goalpost landmark detected at its field-contact point.
    GoalPost,
    /// L-shaped line crossing landmark.
    LSpot,
    /// T-shaped line crossing landmark.
    TSpot,
    /// X-shaped line crossing landmark.
    XSpot,
    /// Penalty marker landmark.
    PenaltySpot,
}

impl VisualFeatureClass {
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::GoalPost => 0,
            Self::LSpot => 1,
            Self::TSpot => 2,
            Self::XSpot => 3,
            Self::PenaltySpot => 4,
        }
    }
}
