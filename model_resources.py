"""Explicit FMA conventions and unsupported-operation reporting."""
import torch


def measure_supported_operations(model,device,img_size):
    """fvcore counts an FMA as one; never divide its total by two for MACs.

    Only the known convolution/linear/matrix contraction counts are MACs.
    Other supported counts retain their own scalar-operation convention.
    FFT, complex products and elementwise/transcendental costs are incomplete.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print('fvcore unavailable; operation profile not fabricated')
        return None
    model.eval();dummy=torch.randn(1,3,img_size,img_size,device=device)
    analysis=FlopCountAnalysis(model,dummy)
    analysis.uncalled_modules_warnings(False)
    total=float(analysis.total())
    operators={str(k):float(v) for k,v in analysis.by_operator().items()}
    unsupported={str(k):int(v) for k,v in analysis.unsupported_ops().items()}
    fma_names={'conv','linear','matmul','bmm','einsum','addmm','mm'}
    macs=sum(v for k,v in operators.items() if k in fma_names)
    supported_arithmetic=2*macs+sum(v for k,v in operators.items() if k not in fma_names)
    return {'flops':supported_arithmetic,'flops_G':supported_arithmetic/1e9,
            'macs':macs,'macs_G':macs/1e9,'fvcore_fma1_supported_total':total,
            'counting_convention':'known real FMA operations count as 2 scalar arithmetic operations; supported only',
            'by_operator_fvcore':operators,'unsupported_operators':unsupported,
            'complete_fft_complex_flops':False}
