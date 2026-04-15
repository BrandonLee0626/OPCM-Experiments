import copy

import torch

from src.task_vector import TaskVector
from src.utils import frobenius_inner_product

# ---------------------------------------------------------------------------
# OPCM
# ---------------------------------------------------------------------------

class OPCM:
    def __init__(self, alpha, task_vector: TaskVector):
        self.alpha = alpha
        self.merged_task_vector = task_vector
        self.merged_task_number = 1
        self.avg_task_vector_norm = task_vector.linear_weight_norm()
        self.scaling_factor = 1
        self.previous_lambda_t = 1.0
        self.projected_task_vector_sum = task_vector
        self.first_tv = task_vector

    def get_split_rank(self, S):
        return (S.cumsum(dim=0) / S.sum() > self.alpha).float().argmax().item()

    def project_linear_weight(self, svd_result, linear_weight, split_rank):
        U, _, V = svd_result

        coeff_mat = U.T @ linear_weight @ V
        coeff_mat.fill_diagonal_(0)
        coeff_mat[:split_rank, :split_rank] = 0

        projected_linear_weight = U @ coeff_mat @ V.T

        return projected_linear_weight

    def project_task_vector(self, tv: TaskVector):
        """Project tv onto the null space of merged_task_vector.

        Returns:
            projected_task_vector: TaskVector
            metrics: dict with inner_product, inner_product_with_first, approx_error, rank
        """
        projected_task_vector = copy.deepcopy(tv)

        svd_result_dict = self.merged_task_vector.svd_linear_weight()

        total_fip = 0.0
        total_fip_w_first = 0.0
        total_error = 0.0
        total_rank = 0
        layer_info = {}

        for linear_weight_name in tv.linear_weight_list:
            svd_result = svd_result_dict[linear_weight_name]
            weight = tv.backbone[linear_weight_name]
            min_dim = min(weight.shape)

            split_rank = self.get_split_rank(svd_result[1])
            total_rank += split_rank

            layer_info[linear_weight_name] = {
                'split_rank': split_rank,
                'min_dim':    min_dim,
            }

            projected_linear_weight = self.project_linear_weight(
                svd_result, weight, split_rank
            )
            projected_task_vector.backbone[linear_weight_name] = projected_linear_weight

            total_fip += frobenius_inner_product(
                self.merged_task_vector.backbone[linear_weight_name], projected_linear_weight
            )
            total_fip_w_first += frobenius_inner_product(
                self.first_tv.backbone[linear_weight_name], projected_linear_weight
            )
            total_error += torch.linalg.norm(
                projected_linear_weight - weight, ord='fro'
            )

        n = len(tv.linear_weight_list)
        metrics = {
            'inner_product': total_fip / n,
            'inner_product_with_first': total_fip_w_first / n,
            'approx_error': total_error / n,
            'rank': total_rank / n,
            'layer_info': layer_info,
        }

        return projected_task_vector, metrics

    def merge_task_vector(self, tv: TaskVector):
        projected_task_vector, metrics = self.project_task_vector(tv)

        numerator_tv = self.previous_lambda_t * self.merged_task_vector + projected_task_vector

        tv_norm = tv.linear_weight_norm()
        merged_task_number = self.get_merged_task_number()
        self.avg_task_vector_norm = (
            (merged_task_number * self.avg_task_vector_norm + tv_norm) / (merged_task_number + 1)
        )

        new_norm = numerator_tv.linear_weight_norm()
        self.lambda_t = new_norm / self.avg_task_vector_norm

        self.merged_task_vector = (1 / self.lambda_t) * numerator_tv
        self.previous_lambda_t = self.lambda_t
        self.merged_task_number += 1

        return metrics

    def get_merged_task_vector(self):
        return self.merged_task_vector

    def get_merged_task_number(self):
        return self.merged_task_number
