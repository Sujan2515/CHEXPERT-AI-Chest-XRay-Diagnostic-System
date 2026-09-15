import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from config import (
    DEVICE,
    PNEUMONIA_MODEL_PATH,
    TB_MODEL_PATH,
)

from dataset import (
    get_pneumonia_datasets,
    get_tb_datasets,
    create_loader,
)

from model import create_model


def evaluate_model(test_dataset, model_path, model_name):

    print("\n" + "=" * 60)
    print(f"EVALUATING {model_name.upper()} MODEL")
    print("=" * 60)

    test_loader = create_loader(
        test_dataset,
        shuffle=False
    )

    model = create_model()

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            _, predictions = torch.max(
                outputs,
                1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp)

    print("\nTest images :", len(test_dataset))

    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1-score    : {f1:.4f}")
    print(f"Specificity : {specificity:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nDetailed counts:")
    print("True Negatives :", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives  :", tp)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":

    print("=" * 60)
    print("CHEXPERT MODEL EVALUATION")
    print("=" * 60)

    print("Device:", DEVICE)

    # ========================================================
    # PNEUMONIA
    # ========================================================

    _, _, pneumonia_test = get_pneumonia_datasets()

    pneumonia_results = evaluate_model(
        test_dataset=pneumonia_test,
        model_path=PNEUMONIA_MODEL_PATH,
        model_name="Pneumonia",
    )

    # ========================================================
    # TB
    # ========================================================

    _, _, tb_test = get_tb_datasets()

    tb_results = evaluate_model(
        test_dataset=tb_test,
        model_path=TB_MODEL_PATH,
        model_name="Tuberculosis",
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)

    print("\nPNEUMONIA")
    print(
        f"Accuracy    : {pneumonia_results['accuracy']:.4f}"
    )
    print(
        f"Precision   : {pneumonia_results['precision']:.4f}"
    )
    print(
        f"Recall      : {pneumonia_results['recall']:.4f}"
    )
    print(
        f"F1-score    : {pneumonia_results['f1']:.4f}"
    )
    print(
        f"Specificity : {pneumonia_results['specificity']:.4f}"
    )

    print("\nTUBERCULOSIS")
    print(
        f"Accuracy    : {tb_results['accuracy']:.4f}"
    )
    print(
        f"Precision   : {tb_results['precision']:.4f}"
    )
    print(
        f"Recall      : {tb_results['recall']:.4f}"
    )
    print(
        f"F1-score    : {tb_results['f1']:.4f}"
    )
    print(
        f"Specificity : {tb_results['specificity']:.4f}"
    )