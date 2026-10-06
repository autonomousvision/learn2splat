import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars("OPT")
os.environ["OPT"] = " ".join(
    flag for flag in opt.split() if flag != "-Wstrict-prototypes"
)

setup(
    name="fused_knn_attn_cuda",
    version="1.0",
    install_requires=["torch"],
    ext_modules=[
        CUDAExtension(
            name="fused_knn_attn_cuda",
            sources=[
                "csrc/fused_knn_attn.cpp",
                "csrc/fused_knn_attn_kernel.cu",
            ],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
