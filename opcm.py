import torch
import copy, math, argparse, mlflow

from src.model import MultiTaskViT
from src.task_vector import TaskVector
from src.utils import load_task_vectors, evaluate_model, frobenius_inner_product

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

        self.rank_count = {
            linear_weight_name: 0
            for linear_weight_name in task_vector.linear_weight_list
        }

    def get_split_rank(self, S):
        return (S.cumsum(dim=0) / S.sum() > self.alpha).float().argmax().item()
    
    def project_linear_weight(self, svd_result, linear_weight, split_rank):
        U, S, V = svd_result

        coeff_mat = U.T @ linear_weight @ V
        coeff_mat.fill_diagonal_(0)
        coeff_mat[:split_rank, :split_rank] = 0
        
        projected_linear_weight = U @ coeff_mat @ V.T

        return projected_linear_weight
    
    def project_task_vector(self, tv: TaskVector):
        projected_task_vector = copy.deepcopy(tv)

        svd_result_dict = self.merged_task_vector.svd_linear_weight()

        total_fip = 0.0
        total_fip_w_first = 0.0
        total_error = 0.0
        total_rank = 0

        for linear_weight_name in tv.linear_weight_list:
            svd_result = svd_result_dict[linear_weight_name]
            
            split_rank = self.get_split_rank(svd_result[1])
            self.rank_count[linear_weight_name] += split_rank
            total_rank += split_rank
            mlflow.log_metric(f"added_split_rank_{linear_weight_name}", self.rank_count[linear_weight_name], step=self.merged_task_number)

            projected_linear_weight = self.project_linear_weight(svd_result, tv.backbone[linear_weight_name], split_rank)

            projected_task_vector.backbone[linear_weight_name] = projected_linear_weight

            total_fip += frobenius_inner_product(self.merged_task_vector.backbone[linear_weight_name], projected_linear_weight)
            total_fip_w_first += frobenius_inner_product(self.first_tv.backbone[linear_weight_name], projected_linear_weight)
            total_error += torch.linalg.norm(projected_linear_weight - tv.backbone[linear_weight_name], ord='fro')
        
        mlflow.log_metric(key='inner product', value=total_fip / len(tv.linear_weight_list), step=self.merged_task_number)
        mlflow.log_metric(key='inner product with first', value=total_fip_w_first / len(tv.linear_weight_list), step=self.merged_task_number)
        mlflow.log_metric(key='approx error', value=total_error / len(tv.linear_weight_list), step=self.merged_task_number)
        mlflow.log_metric(key='rank', value=total_rank / len(tv.linear_weight_list), step=self.merged_task_number)

        return projected_task_vector
    
    def merge_task_vector(self, tv: TaskVector):
        projected_task_vector = self.project_task_vector(tv)

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

    def get_merged_task_vector(self):
        return self.merged_task_vector
    
    def get_merged_task_number(self):
        return self.merged_task_number
    
def main(args):
    alpha = args.alpha

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiTaskViT()
    model.to(device)

    task_vectors = load_task_vectors(device)
    tasks = [tv.trained_task_names[0] for tv in task_vectors]

    opcm = OPCM(alpha, task_vectors[0])

    merged_task_vector = opcm.get_merged_task_vector()
    model.load_task_vector(merged_task_vector)

    test_accuracies = evaluate_model(model, tasks[:1])
    mlflow.log_metrics(test_accuracies, step=1)

    for tv in task_vectors[1:]:
        opcm.merge_task_vector(tv)

        merged_task_vector = opcm.get_merged_task_vector()
        merged_task_number = opcm.get_merged_task_number()

        model.load_task_vector(merged_task_vector)

        test_accuracies = evaluate_model(model, tasks[:merged_task_number])
        mlflow.log_metrics(test_accuracies, step=merged_task_number)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha', type=float, default=0.5)

    args = parser.parse_args()

    mlflow.set_experiment('OPCM_experiments')

    with mlflow.start_run():
        mlflow.log_param('alpha', args.alpha)
 
        main(args)