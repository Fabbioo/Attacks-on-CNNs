from .imports import *
from .attacks import *
from .model import inference
from .utility import *

def compute_accuracy(attack_type: int, dataset: list, model: torchvision.models, device: str, img_resize: tuple, epsilons: list, alphas: list, iters: int = 0) -> tuple[list, dict]:
    
    accuracies: list = []
    dict_wrong_preds: dict = {}
    
    for epsilon, alpha in zip(epsilons, alphas):
        
        correct_predicts: int = 0
        wrong_preds: list = []
        
        for image in dataset:
            
            loss_fn = nn.CrossEntropyLoss()
            
            original_image: torch.Tensor = read_image(image)
            original_image = preprocess(original_image, img_resize).to(device)
            
            if attack_type == 1:
                perturbed_image: torch.Tensor = fgsm_attack(model, loss_fn, original_image, epsilon, device)
            elif attack_type == 2:
                perturbed_image: torch.Tensor = ifgsm_attack(model, loss_fn, original_image, epsilon, alpha, iters, device)
            elif attack_type == 3:
                perturbed_image: torch.Tensor = pgd_attack(model, loss_fn, original_image, epsilon, alpha, iters, device)
            
            original_image = postprocess(original_image)
            perturbed_image = postprocess(perturbed_image)
            pred1: int = inference(model, original_image)[0]
            pred2: int = inference(model, perturbed_image)[0]
            if pred1 == pred2:
                correct_predicts += 1
            else:
                wrong_preds.append(perturbed_image)
        
        correct_predicts /= len(dataset)
        accuracies.append(correct_predicts)
        np.random.shuffle(wrong_preds)
        dict_wrong_preds[epsilon] = wrong_preds
        
    return (accuracies, dict_wrong_preds)