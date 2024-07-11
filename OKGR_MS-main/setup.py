import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import x2ms_adapter

def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run([ 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*x2ms_adapter.tensor_api.split(module, '.'), src) for src in sources]
    )
    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == '__main__':
    # version = '0.6.0+%s' % get_git_commit_number()
    # write_version_to_file(version, 'pcdet/version.py')

    setup(
        name='pcdet',
        version='0.6.0',
        description='OpenPCDet is a general codebase for 3D object detection from point cloud',
        install_requires=[
            'numpy',
            'llvmlite',
            'numba',
            'tensorboardX',
            'easydict',
            'pyyaml',
            'scikit-image',
            'tqdm',
            'SharedArray',
            # 'spconv',  # spconv has different names depending on the cuda version
        ],

        author='Shaoshuai Shi',
        author_email='shaoshuaics@gmail.com',
        license='Apache License 2.0',
        packages=find_packages(exclude=['tools', 'data', 'output']),
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name='iou3d_nms_cuda',
                module='pcdet.ops.iou3d_nms',
                sources=[
                    'src/iou3d_cpu.cpp',
                    'src/iou3d_nms_api.cpp',
                    'src/iou3d_nms.cpp',
                ]
            ),
            make_cuda_ext(
                name='roiaware_pool3d_cuda',
                module='pcdet.ops.roiaware_pool3d',
                sources=[
                    'src/roiaware_pool3d.cpp',
                ]
            ),
            make_cuda_ext(
                name='roipoint_pool3d_cuda',
                module='pcdet.ops.roipoint_pool3d',
                sources=[
                    'src/roipoint_pool3d.cpp',
                ]
            ),
            make_cuda_ext(
                name='pointnet2_stack_cuda',
                module='pcdet.ops.pointnet2.pointnet2_stack',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/group_points.cpp',
                    'src/sampling.cpp',
                    'src/interpolate.cpp',
                    'src/voxel_query.cpp',
                    'src/vector_pool.cpp',
                ],
            ),
            make_cuda_ext(
                name='pointnet2_batch_cuda',
                module='pcdet.ops.pointnet2.pointnet2_batch',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/group_points.cpp',
                    'src/interpolate.cpp'
                    'src/sampling.cpp',

                ],
            ),
        ],
    )
