from .imports import torchvision, plt, np, grad_cam
from .model import inference
from .utility import tensor2ndarray

def preds_display(model: torchvision.models, tripla: tuple, epsilon: float, show_noise: bool = False) -> None:
    
    if show_noise:
        images: list = [tensor2ndarray(tripla[0]), tensor2ndarray(tripla[1]), tensor2ndarray(tripla[2])]
        objects: list = ['ORIGINAL', 'NOISE', 'PERTURBED']
    else:
        images: list = [tensor2ndarray(tripla[0]), tensor2ndarray(tripla[2])]
        objects: list = ['ORIGINAL', 'PERTURBED']
    
    outputs_orig: tuple = inference(model, tripla[0])
    outputs_pert: tuple = inference(model, tripla[2])
    if outputs_orig[1] == outputs_pert[1]: # Se le due predizioni coincidono ...
        color: str = 'green' # ... stampo una scritta verde ...
    else:
        color: str = 'red' # ... altrimenti rossa.
    
    plt.figure()
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        if i == 0:
            plt.title(objects[0] + '\n\n' + str(outputs_orig[0]) + ': ' + outputs_orig[1] + f' -> {outputs_orig[2] * 100:.3}%', color = 'green')
        elif i == 1 and show_noise:
            plt.title(objects[1])
        elif i == 1 and not show_noise:
            plt.title(objects[1] + f' (Epsilon: {epsilon})\n\n' + str(outputs_pert[0]) + ': ' + outputs_pert[1] + f' -> {outputs_pert[2] * 100:.3}%', color = color)
        else:
            plt.title(objects[2] + f' (Epsilon: {epsilon})\n\n' + str(outputs_pert[0]) + ': ' + outputs_pert[1] + f' -> {outputs_pert[2] * 100:.3}%', color = color)

def gradcam_display(model: torchvision.models, tripla: tuple, resize: tuple) -> None:
    
    layer: str = 'layer4'
    
    titles: list = ['ORIGINAL', 'PERTURBED']
    
    outputs_orig: tuple = inference(model, tripla[0])
    outputs_pert: tuple = inference(model, tripla[2])
    if outputs_orig[1] == outputs_pert[1]: # Se le due predizioni coincidono ...
        color: str = 'green' # ... stampo una scritta verde ...
    else:
        color: str = 'red' # ... altrimenti rossa.
    
    cam_orig = grad_cam(model, tripla[0], target = outputs_orig[0], saliency_layer = layer)
    cam_orig = (cam_orig - cam_orig.min()) / (cam_orig.max() - cam_orig.min())
    cam_orig = torchvision.transforms.functional.resize(cam_orig, [resize[0], resize[1]])
    image_to_show_orig = cam_orig[0].permute(1, 2, 0).detach().cpu().numpy()
    
    cam_pert = grad_cam(model, tripla[2], target = outputs_pert[0], saliency_layer = layer)
    cam_pert = (cam_pert - cam_pert.min()) / (cam_pert.max() - cam_pert.min())
    cam_pert = torchvision.transforms.functional.resize(cam_pert, [resize[0], resize[1]])
    image_to_show_pert = cam_pert[0].permute(1, 2, 0).detach().cpu().numpy()
    
    plt.figure()
    for i in range(len(titles)):
        plt.subplot(1, len(titles), i + 1)
        plt.axis('off')
        if i == 0:
            plt.imshow(image_to_show_orig)
            plt.imshow(tensor2ndarray(tripla[0]), alpha = 0.4)
            plt.title(titles[i] + '\n\n' + str(outputs_orig[0]) + ': ' + outputs_orig[1] + f' -> {outputs_orig[2] * 100:.3}%', color = 'green')
        else:
            plt.imshow(image_to_show_pert)
            plt.imshow(tensor2ndarray(tripla[2]), alpha = 0.4)
            plt.title(titles[i] + '\n\n' + str(outputs_pert[0]) + ': ' + outputs_pert[1] + f' -> {outputs_pert[2] * 100:.3}%', color = color)

def accuracy_display(dataset: list, model: torchvision.models, epsilons: list, accuracies: tuple, iter: int, wrong_preds: tuple, dict_show_wrong_preds: dict) -> None:
    
    plt.figure()
    plt.plot(epsilons, accuracies[0], label = 'FGSM', marker = 'o', color = 'r')
    plt.plot(epsilons, accuracies[1], label = 'I-FGSM', marker = 'o', color = 'g')
    plt.plot(epsilons, accuracies[2], label = 'PGD', marker = 'o', color = 'b')
    plt.legend(loc = 'lower left')
    plt.suptitle(f'Performance del modello ResNet-50 al variare di epsilon su un dataset di {len(dataset)} immagini')
    plt.title(f'Iters = {iter}')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, 1.1, step = 0.1))
    plt.yticks(np.arange(0, 1.1, step = 0.1))
    plt.grid()
    
    temp_str: str = ''
    if dict_show_wrong_preds['show_FGSM_wrong_preds'] == True:
        temp_str = 'FGSM ->'
        wrong_preds_display(wrong_preds[0], dataset, model, epsilons, temp_str)
    if dict_show_wrong_preds['show_IFGSM_wrong_preds'] == True:
        temp_str = f'I-FGSM ->'
        wrong_preds_display(wrong_preds[1], dataset, model, epsilons, temp_str)
    if dict_show_wrong_preds['show_PGD_wrong_preds'] == True:
        temp_str = f'PGD ->'
        wrong_preds_display(wrong_preds[2], dataset, model, epsilons, temp_str)

def wrong_preds_display(dict_wrong_preds: dict, dataset: list, model: torchvision.models, epsilons: list, temp_str: str) -> None:
    
    column_number: int = 4 # Numero arbitrario di grafici da creare per ciascun valore di epsilon.
    
    # Se non ci sono abbastanza grafici da creare per il particolare valore di epsilon modifico il numero di grafici da creare.
    min_number_of_elements_in_list_for_each_epsilon: int = len(dataset) # Scelta arbitraria di inizializzazione.
    for i in range(len(epsilons)):
        if epsilons[i] == 0: # In corrispondenza di epsilon = 0 non ci possono essere errori nelle predizioni, perciò passo direttamente all'iterazione successiva.
            continue
        len_dict_wrong_preds_epsilons_i: int = len(dict_wrong_preds[epsilons[i]])
        if len_dict_wrong_preds_epsilons_i < min_number_of_elements_in_list_for_each_epsilon:
            min_number_of_elements_in_list_for_each_epsilon = len_dict_wrong_preds_epsilons_i
    if column_number > min_number_of_elements_in_list_for_each_epsilon:
        column_number = min_number_of_elements_in_list_for_each_epsilon
    
    # Plot delle immagini perturbate classificate erroneamente.
    for i in range(len(epsilons)):
        if epsilons[i] == 0: # In corrispondenza di epsilon = 0 non ci possono essere errori nelle predizioni, perciò passo direttamente all'iterazione successiva.
            continue
        plt.figure()
        for j in range(column_number):
            _, class_name, class_conf = inference(model, dict_wrong_preds[epsilons[i]][j])
            plt.suptitle(f'{temp_str} Epsilon: {epsilons[i]}')
            plt.subplot(1, column_number, j + 1)
            plt.imshow(tensor2ndarray(dict_wrong_preds[epsilons[i]][j]))
            plt.title(f'Wrong pred:\n' + class_name + '\n' + f'{class_conf*100:.3}%' + '\n', color = 'red')
            plt.axis('off')