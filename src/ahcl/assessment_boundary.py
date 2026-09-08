"""Compose receipt, predictor, and durable attempt checks before row access.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
This narrow model-free boundary is not yet the model loader entrypoint.
"""
from .assessment_custody import claim_assessment_attempt
from .sealed_predictor import SealedLinearPredictor
from .verified_review import verify_launch_receipt


def assess_once(*, packet, config, receipt, registry, operator_id, custody_root,
                predictor, evaluate):
    """Consume the exact reviewed assessment allocation before its callback.

    The host owns evaluate and registry; neither may come from model output.
    evaluate receives only the parameter-bound predictor. Its source identity
    and raw custody behavior must be independently reviewed before real use.
    """
    if type(predictor) is not SealedLinearPredictor:
        raise ValueError('only the fixed parameter-bound predictor is supported')
    if config.get('execution', {}).get('enabled') is not True:
        raise ValueError('execution is disabled')
    expected = config.get('assessment', {}).get('predictor_digest')
    if expected != predictor.identity:
        raise ValueError('predictor is not bound to reviewed assessment config')
    verify_launch_receipt(packet, config, receipt, registry,
                         operator_id=operator_id, purpose='assessment')
    claim_assessment_attempt(custody_root, packet_id=packet['packet_digest'],
                             predictor_id=predictor.identity)
    return evaluate(predictor)
