import torch
import math
from . import pointnet2_stack_cuda as pointnet2
import time
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn
class BallQuery(x2ms_adapter.autograd.Function):

    @staticmethod
    def construct(ctx, radius: float, nsample: int, xyz, xyz_batch_cnt,
                new_xyz, new_xyz_batch_cnt):
        """
        Args:
            ctx:
            radius: float, radius of the balls
            nsample: int, maximum number of features in the balls
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]

        Returns:
            idx: (M1 + M2, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert new_xyz_batch_cnt.is_contiguous()
        assert xyz.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()
        #print('pointstack.BallQuery')
        B = xyz_batch_cnt.shape[0]
        M = new_xyz.shape[0]
        idx = x2ms_adapter.tensor_api.zero_(x2ms_adapter.IntTensor(M, nsample))

        pointnet2.ball_query_wrapper(B, M, radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx)
        empty_ball_mask = (idx[:, 0] == -1)
        idx[empty_ball_mask] = 0

        ctx.mark_non_differentiable(idx)
        ctx.mark_non_differentiable(empty_ball_mask)

        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None, None, None, None, None


ball_query = BallQuery.apply


class GroupingOperation(x2ms_adapter.autograd.Function):

    @staticmethod
    def construct(ctx, features, features_batch_cnt,
                idx, idx_batch_cnt):
        """
        Args:
            ctx:
            features: (N1 + N2 ..., C) tensor of features to group
            features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
            idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
            idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with

        Returns:
            output: (M1 + M2, C, nsample) tensor
        """
        #print('pointstack.GroupingOperation')
        assert features.is_contiguous()
        assert features_batch_cnt.is_contiguous()
        assert idx.is_contiguous()
        assert idx_batch_cnt.is_contiguous()

        assert features.shape[0] == x2ms_adapter.tensor_api.x2ms_sum(features_batch_cnt), \
            'features: %s, features_batch_cnt: %s' % (str(features.shape), str(features_batch_cnt))
        assert idx.shape[0] == x2ms_adapter.tensor_api.x2ms_sum(idx_batch_cnt), \
            'idx: %s, idx_batch_cnt: %s' % (str(idx.shape), str(idx_batch_cnt))

        M, nsample = x2ms_adapter.tensor_api.x2ms_size(idx)
        N, C = x2ms_adapter.tensor_api.x2ms_size(features)
        B = idx_batch_cnt.shape[0]
        output = x2ms_adapter.FloatTensor(M, C, nsample)

        pointnet2.group_points_wrapper(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, output)

        ctx.for_backwards = (B, N, idx, features_batch_cnt, idx_batch_cnt)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        """
        Args:
            ctx:
            grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward

        Returns:
            grad_features: (N1 + N2 ..., C) gradient of the features
        """
        B, N, idx, features_batch_cnt, idx_batch_cnt = ctx.for_backwards

        M, C, nsample = x2ms_adapter.tensor_api.x2ms_size(grad_out)
        grad_features = x2ms_adapter.autograd.Variable(x2ms_adapter.tensor_api.zero_(x2ms_adapter.FloatTensor(N, C)))

        grad_out_data = x2ms_adapter.tensor_api.contiguous(grad_out.data)
        pointnet2.group_points_grad_wrapper(B, M, C, N, nsample, grad_out_data, idx,
                                            idx_batch_cnt, features_batch_cnt, grad_features.data)
        return grad_features, None, None, None


grouping_operation = GroupingOperation.apply


class QueryAndGroup(nn.Cell):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        Args:
            radius: float, radius of ball
            nsample: int, maximum number of features to gather in the ball
            use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def construct(self, xyz, xyz_batch_cnt,
                new_xyz, new_xyz_batch_cnt,
                features = None):
        """
        Args:
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            features: (N1 + N2 ..., C) tensor of features to group

        Returns:
            new_features: (M1 + M2, C, nsample) tensor
        """
        #print('pointstack.QueryAndGroup')
        assert xyz.shape[0] == x2ms_adapter.tensor_api.x2ms_sum(xyz_batch_cnt), 'xyz: %s, xyz_batch_cnt: %s' % (str(xyz.shape), str(new_xyz_batch_cnt))
        assert new_xyz.shape[0] == x2ms_adapter.tensor_api.x2ms_sum(new_xyz_batch_cnt), \
            'new_xyz: %s, new_xyz_batch_cnt: %s' % (str(new_xyz.shape), str(new_xyz_batch_cnt))

        # idx: (M1 + M2 ..., nsample), empty_ball_mask: (M1 + M2 ...)
        idx, empty_ball_mask = ball_query(self.radius, self.nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt)
        grouped_xyz = grouping_operation(xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, 3, nsample)
        grouped_xyz -= x2ms_adapter.tensor_api.unsqueeze(new_xyz, -1)

        grouped_xyz[empty_ball_mask] = 0

        if features is not None:
            grouped_features = grouping_operation(features, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, C, nsample)
            grouped_features[empty_ball_mask] = 0
            if self.use_xyz:
                new_features = x2ms_adapter.cat([grouped_xyz, grouped_features], dim=1)  # (M1 + M2 ..., C + 3, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features, idx


class FarthestPointSampling(x2ms_adapter.autograd.Function):
    @staticmethod
    def construct(ctx, xyz, npoint: int):
        """
        Args:
            ctx:
            xyz: (B, N, 3) where N > npoint
            npoint: int, number of features in the sampled set

        Returns:
            output: (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        #print('pointstack.FarthestPointSampling')
        B, N, _ = x2ms_adapter.tensor_api.x2ms_size(xyz)
        output = x2ms_adapter.IntTensor(B, npoint)
        temp = x2ms_adapter.tensor_api.fill_(x2ms_adapter.FloatTensor(B, N), 1e10)

        pointnet2.farthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


farthest_point_sample = furthest_point_sample = FarthestPointSampling.apply

class StackFarthestPointSampling(x2ms_adapter.autograd.Function):
    @staticmethod
    def construct(ctx, xyz, xyz_batch_cnt, npoint):
        """
        Args:
            ctx:
            xyz: (N1 + N2 + ..., 3) where N > npoint
            xyz_batch_cnt: [N1, N2, ...]
            npoint: int, number of features in the sampled set

        Returns:
            output: (npoint.sum()) tensor containing the set,
            npoint: (M1, M2, ...)
        """
        batch_size = xyz_batch_cnt.__len__()
        if not isinstance(npoint, mindspore.Tensor):
            if not isinstance(npoint, list):
                npoint = [npoint for i in range(batch_size)]
            npoint = mindspore.tensor(npoint).int()

        N, _ = xyz.shape
        xyz=torch.Tensor(xyz.asnumpy())
        xyz_batch_cnt=torch.Tensor(xyz_batch_cnt.asnumpy())
        npoint=torch.Tensor(npoint.asnumpy()).int()
        temp = torch.FloatTensor(N).fill_(1e10)
        output = torch.IntTensor(npoint.sum().item())

        out=pointnet2.stack_farthest_point_sampling_wrapper(xyz.cuda(), temp.cuda(), xyz_batch_cnt.int().cuda(), output.int().cuda(), npoint.int().cuda())
        
        return mindspore.Tensor(out.cpu().numpy())

    @staticmethod
    def backward(xyz, a=None):
        return None, None


stack_farthest_point_sample = StackFarthestPointSampling.apply


class ThreeNN(x2ms_adapter.autograd.Function):
    @staticmethod
    def construct(ctx, unknown, unknown_batch_cnt, known, known_batch_cnt):
        """
        Args:
            ctx:
            unknown: (N1 + N2..., 3)
            unknown_batch_cnt: (batch_size), [N1, N2, ...]
            known: (M1 + M2..., 3)
            known_batch_cnt: (batch_size), [M1, M2, ...]

        Returns:
            dist: (N1 + N2 ..., 3)  l2 distance to the three nearest neighbors
            idx: (N1 + N2 ..., 3)  index of the three nearest neighbors, range [0, M1+M2+...]
        """
        #print('pointstack.ThreeNN')
        assert unknown.shape.__len__() == 2 and unknown.shape[1] == 3
        assert known.shape.__len__() == 2 and known.shape[1] == 3
        assert unknown_batch_cnt.__len__() == known_batch_cnt.__len__()

        dist2 = x2ms_adapter.tensor_api.new_zeros(unknown, unknown.shape)
        idx = x2ms_adapter.tensor_api.x2ms_int(x2ms_adapter.tensor_api.new_zeros(unknown_batch_cnt, unknown.shape))

        pointnet2.three_nn_wrapper(
            x2ms_adapter.tensor_api.contiguous(unknown), x2ms_adapter.tensor_api.contiguous(unknown_batch_cnt),
            x2ms_adapter.tensor_api.contiguous(known), x2ms_adapter.tensor_api.contiguous(known_batch_cnt), dist2, idx
        )
        return x2ms_adapter.sqrt(dist2), idx
    
    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(x2ms_adapter.autograd.Function):

    @staticmethod
    def construct(ctx,features,idx, weight):
        """
        Args:
            ctx:
            features: (M1 + M2 ..., C)
            idx: [N1 + N2 ..., 3]
            weight: [N1 + N2 ..., 3]

        Returns:
            out_tensor: (N1 + N2 ..., C)
        """
        # print('pointstack.ThreeInterpolate')
        assert idx.shape[0] == weight.shape[0] and idx.shape[1] == weight.shape[1] == 3

        # ctx.three_interpolate_for_backward = (idx, weight, features.shape[0])
        # output = x2ms_adapter.tensor_api.new_zeros(features, (idx.shape[0], features.shape[1]))
        features=torch.Tensor(features.asnumpy())
        output = features.new_zeros((idx.shape[0], features.shape[1]))
        idx=torch.Tensor(idx.asnumpy()).int()
        weight=torch.Tensor(weight.asnumpy())
        # output=torch.Tensor(output.asnumpy())
        out=pointnet2.three_interpolate_wrapper(features.cuda(), idx.cuda(), weight.cuda(), output.cuda())
        
        out=out.cpu()
        output=mindspore.tensor(out.numpy())
        return output

    @staticmethod
    def backward(ctx, grad_out):
        """
        Args:
            ctx:
            grad_out: (N1 + N2 ..., C)

        Returns:
            grad_features: (M1 + M2 ..., C)
        """
        idx, weight, M = ctx.three_interpolate_for_backward
        grad_features = x2ms_adapter.tensor_api.new_zeros(grad_out, (M, grad_out.shape[1]))
        pointnet2.three_interpolate_grad_wrapper(
            x2ms_adapter.tensor_api.contiguous(grad_out), x2ms_adapter.tensor_api.contiguous(idx), x2ms_adapter.tensor_api.contiguous(weight), grad_features
        )
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply

class ThreeNNForVectorPoolByTwoStep(x2ms_adapter.autograd.Function):
    @staticmethod
    def construct(ctx, support_xyz, xyz_batch_cnt, new_xyz, new_xyz_grid_centers, new_xyz_batch_cnt,
                max_neighbour_distance, nsample, neighbor_type, avg_length_of_neighbor_idxs, num_total_grids,
                neighbor_distance_multiplier):
        """
        Args:
            ctx:
            // support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            // xyz_batch_cnt: (batch_size), [N1, N2, ...]
            // new_xyz: (M1 + M2 ..., 3) centers of the ball query
            // new_xyz_grid_centers: (M1 + M2 ..., num_total_grids, 3) grids centers of each grid
            // new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            // nsample: find all (-1), find limited number(>0)
            // neighbor_type: 1: ball, others: cube
            // neighbor_distance_multiplier: query_distance = neighbor_distance_multiplier * max_neighbour_distance

        Returns:
            // new_xyz_grid_idxs: (M1 + M2 ..., num_total_grids, 3) three-nn
            // new_xyz_grid_dist2: (M1 + M2 ..., num_total_grids, 3) square of dist of three-nn
        """
        # print('pointstack.ThreeNNForVectorPoolByTwoStep')
        num_new_xyz = new_xyz.shape[0]
        new_xyz_grid_dist2 = new_xyz_grid_centers.new_zeros(new_xyz_grid_centers.shape)
        new_xyz_grid_idxs = new_xyz_grid_centers.new_zeros(new_xyz_grid_centers.shape).int()

        while True:
            num_max_sum_points = avg_length_of_neighbor_idxs * num_new_xyz
            stack_neighbor_idxs = new_xyz_grid_idxs.new_zeros(num_max_sum_points)
            start_len = new_xyz_grid_idxs.new_zeros((num_new_xyz, 2)).int()
            cumsum = new_xyz_grid_idxs.new_zeros(1)
            support_xyz=torch.Tensor(support_xyz.asnumpy())
            xyz_batch_cnt=torch.Tensor(xyz_batch_cnt.asnumpy()).int()
            new_xyz=torch.Tensor(new_xyz.asnumpy())
            new_xyz_batch_cnt=torch.Tensor(new_xyz_batch_cnt.asnumpy()).int()
            stack_neighbor_idxs=torch.Tensor(stack_neighbor_idxs.asnumpy()).int()
            start_len=torch.Tensor(start_len.asnumpy()).int()
            cumsum=torch.Tensor(cumsum.asnumpy()).int()
            out=pointnet2.query_stacked_local_neighbor_idxs_wrapper_stack(
                support_xyz.cuda(), xyz_batch_cnt.cuda(),
                new_xyz.cuda(), new_xyz_batch_cnt.cuda(),
                stack_neighbor_idxs.cuda(),start_len.cuda(), cumsum.cuda(),
                avg_length_of_neighbor_idxs, max_neighbour_distance * neighbor_distance_multiplier,
                nsample, neighbor_type
            )
            cumsum,stack_neighbor_idxs,start_len=out[:3]
            avg_length_of_neighbor_idxs = cumsum[0].item() // num_new_xyz + int(cumsum[0].item() % num_new_xyz > 0)

            if cumsum[0].item() <= num_max_sum_points:
                break

        stack_neighbor_idxs = stack_neighbor_idxs[:cumsum[0]]
        new_xyz_grid_centers=torch.Tensor(new_xyz_grid_centers.asnumpy())
        new_xyz_grid_idxs=torch.Tensor(new_xyz_grid_idxs.asnumpy()).int().fill_(-1)
        new_xyz_grid_dist2=torch.Tensor(new_xyz_grid_dist2.asnumpy()).float()
        out1=pointnet2.query_three_nn_by_stacked_local_idxs_wrapper_stack(
            support_xyz.cuda(), new_xyz.cuda(), new_xyz_grid_centers.cuda(), new_xyz_grid_idxs.cuda(), new_xyz_grid_dist2.cuda(),
            stack_neighbor_idxs.cuda(), start_len.cuda(), num_new_xyz, num_total_grids
        )
        new_xyz_grid_dist2,new_xyz_grid_idxs=out1[:2]
        new_xyz_grid_dist2=mindspore.Tensor(new_xyz_grid_dist2.cpu().numpy())
        new_xyz_grid_idxs=mindspore.Tensor(new_xyz_grid_idxs.cpu().numpy())
        # avg_length_of_neighbor_idxs=mindspore.Tensor(avg_length_of_neighbor_idxs)
        return x2ms_adapter.sqrt(new_xyz_grid_dist2), new_xyz_grid_idxs, mindspore.Tensor(avg_length_of_neighbor_idxs)


three_nn_for_vector_pool_by_two_step = ThreeNNForVectorPoolByTwoStep.apply


class VectorPoolWithVoxelQuery(x2ms_adapter.autograd.Function):
    @staticmethod
    def construct(ctx, support_xyz, xyz_batch_cnt, support_features,
                new_xyz, new_xyz_batch_cnt, num_grid_x, num_grid_y, num_grid_z,
                max_neighbour_distance, num_c_out_each_grid, use_xyz,
                num_mean_points_per_grid=100, nsample=-1, neighbor_type=0, pooling_type=0):
        """
        Args:
            ctx:
            support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            support_features: (N1 + N2 ..., C)
            new_xyz: (M1 + M2 ..., 3) centers of new positions
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            num_grid_x: number of grids in each local area centered at new_xyz
            num_grid_y:
            num_grid_z:
            max_neighbour_distance:
            num_c_out_each_grid:
            use_xyz:
            neighbor_type: 1: ball, others: cube:
            pooling_type: 0: avg_pool, 1: random choice
        Returns:
            new_features: (M1 + M2 ..., num_c_out)
        """
        # print('pointstack.VectorPoolWithVoxelQuery')
        # assert support_xyz.is_contiguous()
        # assert support_features.is_contiguous()
        # assert xyz_batch_cnt.is_contiguous()
        # assert new_xyz.is_contiguous()
        # assert new_xyz_batch_cnt.is_contiguous()
        num_total_grids = num_grid_x * num_grid_y * num_grid_z
        num_c_out = num_c_out_each_grid * num_total_grids
        N, num_c_in = support_features.shape
        M = new_xyz.shape[0]

        assert num_c_in % num_c_out_each_grid == 0, \
            f'the input channels ({num_c_in}) should be an integral multiple of num_c_out_each_grid({num_c_out_each_grid})'

        while True:
            new_features = x2ms_adapter.tensor_api.new_zeros(support_features, (M, num_c_out))
            new_local_xyz = x2ms_adapter.tensor_api.new_zeros(support_features, (M, 3 * num_total_grids))
            point_cnt_of_grid = x2ms_adapter.tensor_api.new_zeros(xyz_batch_cnt, (M, num_total_grids))

            num_max_sum_points = num_mean_points_per_grid * M
            grouped_idxs = x2ms_adapter.tensor_api.new_zeros(xyz_batch_cnt, (num_max_sum_points, 3))
            support_xyz=torch.Tensor(support_xyz.asnumpy())
            xyz_batch_cnt=torch.Tensor(xyz_batch_cnt.asnumpy()).int()
            support_features=torch.Tensor(support_features.asnumpy())
            new_xyz=torch.Tensor(new_xyz.asnumpy())
            new_xyz_batch_cnt=torch.Tensor(new_xyz_batch_cnt.asnumpy()).int()
            new_features=torch.Tensor(new_features.asnumpy())
            new_local_xyz=torch.Tensor(new_local_xyz.asnumpy())
            point_cnt_of_grid=torch.Tensor(point_cnt_of_grid.asnumpy()).int()
            grouped_idxs=torch.Tensor(grouped_idxs.asnumpy()).int()
            out = pointnet2.vector_pool_wrapper(
                support_xyz.cuda(), xyz_batch_cnt.cuda(), support_features.cuda(), new_xyz.cuda(), new_xyz_batch_cnt.cuda(),
                new_features.cuda(), new_local_xyz.cuda(), point_cnt_of_grid.cuda(), grouped_idxs.cuda(),
                num_grid_x, num_grid_y, num_grid_z, max_neighbour_distance, use_xyz,
                num_max_sum_points, nsample, neighbor_type, pooling_type
            )
            num_cum_sum,grouped_idxs,point_cnt_of_grid ,new_features,new_local_xyz=out[:5]
            num_mean_points_per_grid = num_cum_sum // M + int(num_cum_sum % M > 0)
            if num_cum_sum <= num_max_sum_points:
                break
        grouped_idxs = grouped_idxs[:num_cum_sum]
        # normalizer = torch.clamp_min(x2ms_adapter.tensor_api.x2ms_float(point_cnt_of_grid[:, :, None]), min=1e-6)
        normalizer = torch.clamp_min(point_cnt_of_grid.cpu()[:, :, None].float(), min=1e-6)
        new_features = (new_features.cpu().view(-1, num_total_grids, num_c_out_each_grid) / normalizer).view(-1, num_c_out)

        if use_xyz:
            new_local_xyz = (new_local_xyz.cpu().view(-1, num_total_grids, 3) / normalizer).view(-1, num_total_grids * 3)
        # num_mean_points_per_grid = torch.Tensor([num_mean_points_per_grid]).int()
        # nsample = torch.Tensor([nsample]).int()
        return mindspore.tensor(new_features.cpu().numpy()), mindspore.tensor(new_local_xyz.cpu().numpy()), mindspore.tensor(num_mean_points_per_grid.numpy()), mindspore.tensor(point_cnt_of_grid.cpu().numpy())

    @staticmethod
    def backward(ctx, grad_new_features, grad_local_xyz, grad_num_cum_sum, grad_point_cnt_of_grid):
        """
        Args:
            ctx:
            grad_new_features: (M1 + M2 ..., num_c_out), num_c_out = num_c_out_each_grid * num_total_grids

        Returns:
            grad_support_features: (N1 + N2 ..., C_in)
        """
        point_cnt_of_grid, grouped_idxs, N, num_c_in = ctx.vector_pool_for_backward
        grad_support_features = x2ms_adapter.tensor_api.new_zeros(grad_new_features, (N, num_c_in))

        if grouped_idxs.shape[0] > 0:
            pointnet2.vector_pool_grad_wrapper(
                x2ms_adapter.tensor_api.contiguous(grad_new_features), point_cnt_of_grid, grouped_idxs,
                grad_support_features
            )

        return None, None, grad_support_features, None, None, None, None, None, None, None, None, None, None, None, None


vector_pool_with_voxel_query_op = VectorPoolWithVoxelQuery.apply


if __name__ == '__main__':
    pass
