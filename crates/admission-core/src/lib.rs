//! Dependency-free admission policy for the clean-room experiment.
//!
//! State slice: aligned-holistic-continual-learning-clean-room-v1.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Proposal,
    Shadow,
    Validated,
    Review,
    Canopy,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Quarantined,
    Admitted,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Evidence {
    pub digest: String,
    pub independent: bool,
    pub prediction_locked: bool,
    pub held_out: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub base_digest: String,
    pub candidate_digest: String,
    pub evidence: Option<Evidence>,
    pub stage: Stage,
}

/// Evaluate whether a candidate may receive authority.
///
/// The function is pure: it records no state and grants no capability. A
/// candidate remains quarantined until it reaches review with independent,
/// prediction-locked, held-out evidence. Closed candidates are rejected.
pub fn evaluate(candidate: &Candidate) -> Decision {
    if candidate.stage == Stage::Closed || candidate.base_digest == candidate.candidate_digest {
        return Decision::Rejected;
    }

    let Some(evidence) = &candidate.evidence else {
        return Decision::Quarantined;
    };

    if evidence.digest.is_empty()
        || !evidence.independent
        || !evidence.prediction_locked
        || !evidence.held_out
    {
        return Decision::Quarantined;
    }

    if candidate.stage == Stage::Review {
        Decision::Admitted
    } else {
        Decision::Quarantined
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(stage: Stage, evidence: Option<Evidence>) -> Candidate {
        Candidate {
            base_digest: "base".into(),
            candidate_digest: "candidate".into(),
            evidence,
            stage,
        }
    }

    fn accepted_evidence() -> Evidence {
        Evidence {
            digest: "packet".into(),
            independent: true,
            prediction_locked: true,
            held_out: true,
        }
    }

    #[test]
    fn no_evidence_is_quarantined() {
        assert_eq!(
            evaluate(&candidate(Stage::Review, None)),
            Decision::Quarantined
        );
    }

    #[test]
    fn incomplete_evidence_is_quarantined() {
        let mut evidence = accepted_evidence();
        evidence.held_out = false;
        assert_eq!(
            evaluate(&candidate(Stage::Review, Some(evidence))),
            Decision::Quarantined
        );
    }

    #[test]
    fn matching_base_and_candidate_is_rejected() {
        let mut item = candidate(Stage::Review, Some(accepted_evidence()));
        item.candidate_digest = item.base_digest.clone();
        assert_eq!(evaluate(&item), Decision::Rejected);
    }

    #[test]
    fn valid_review_is_admitted() {
        assert_eq!(
            evaluate(&candidate(Stage::Review, Some(accepted_evidence()))),
            Decision::Admitted
        );
    }

    #[test]
    fn valid_evidence_before_review_stays_quarantined() {
        assert_eq!(
            evaluate(&candidate(Stage::Validated, Some(accepted_evidence()))),
            Decision::Quarantined
        );
    }

    #[test]
    fn closed_candidate_is_rejected() {
        assert_eq!(
            evaluate(&candidate(Stage::Closed, Some(accepted_evidence()))),
            Decision::Rejected
        );
    }
}
