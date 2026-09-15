import torch
import torch.optim as optim

from config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    PNEUMONIA_MODEL_PATH,
    TB_MODEL_PATH,
)

from dataset import (
    get_pneumonia_datasets,
    get_tb_datasets,
    create_loader,
)

from model import (
    create_model,
    create_loss_function,
)


def train_model(
    train_dataset,
    val_dataset,
    model_path,
    model_name
):

    print("\n" + "=" * 60)
    print(f"TRAINING {model_name.upper()} MODEL")
    print("=" * 60)

    # -------------------------------------------------
    # Data loaders
    # -------------------------------------------------

    train_loader = create_loader(
        train_dataset,
        shuffle=True
    )

    val_loader = create_loader(
        val_dataset,
        shuffle=False
    )

    print("Training images  :", len(train_dataset))
    print("Validation images:", len(val_dataset))

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = create_model()
    model = model.to(DEVICE)

    # -------------------------------------------------
    # Loss function
    # -------------------------------------------------

    criterion = create_loss_function(
        train_dataset
    )

    criterion = criterion.to(DEVICE)

    # -------------------------------------------------
    # Optimizer
    # -------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -------------------------------------------------
    # Best validation loss
    # -------------------------------------------------

    best_val_loss = float("inf")

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------

    for epoch in range(NUM_EPOCHS):

        # =========================
        # Training
        # =========================

        model.train()

        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(
                outputs,
                labels
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            # Statistics
            running_train_loss += (
                loss.item() * images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            total_train += labels.size(0)

            correct_train += (
                predicted == labels
            ).sum().item()

        train_loss = (
            running_train_loss /
            total_train
        )

        train_accuracy = (
            correct_train /
            total_train
        )

        # =========================
        # Validation
        # =========================

        model.eval()

        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                running_val_loss += (
                    loss.item() * images.size(0)
                )

                _, predicted = torch.max(
                    outputs,
                    1
                )

                total_val += labels.size(0)

                correct_val += (
                    predicted == labels
                ).sum().item()

        val_loss = (
            running_val_loss /
            total_val
        )

        val_accuracy = (
            correct_val /
            total_val
        )

        # =========================
        # Print results
        # =========================

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_accuracy:.4f}"
        )

        # =========================
        # Save best model
        # =========================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                model_path
            )

            print(
                f"  ✓ Best model saved → {model_path}"
            )

    print(
        f"\nFinished training {model_name}."
    )

    print(
        f"Best model saved at:\n{model_path}"
    )


def main():

    print("=" * 60)
    print("CHEXPERT MODEL TRAINING")
    print("=" * 60)

    print("Device:", DEVICE)

    # =================================================
    # Pneumonia
    # =================================================

    pneumonia_train, pneumonia_val, pneumonia_test = (
        get_pneumonia_datasets()
    )

    train_model(
        train_dataset=pneumonia_train,
        val_dataset=pneumonia_val,
        model_path=PNEUMONIA_MODEL_PATH,
        model_name="Pneumonia"
    )

    # =================================================
    # Tuberculosis
    # =================================================

    tb_train, tb_val, tb_test = (
        get_tb_datasets()
    )

    train_model(
        train_dataset=tb_train,
        val_dataset=tb_val,
        model_path=TB_MODEL_PATH,
        model_name="Tuberculosis"
    )


if __name__ == "__main__":

    print("=" * 60)
    print("CHEXPERT MODEL TRAINING")
    print("=" * 60)

    print("Device:", DEVICE)

    # ========================================================
    # TRAIN PNEUMONIA MODEL
    # ========================================================

    pneumonia_train, pneumonia_val, pneumonia_test = (
        get_pneumonia_datasets()
    )

    train_model(
        train_dataset=pneumonia_train,
        val_dataset=pneumonia_val,
        model_path=PNEUMONIA_MODEL_PATH,
        model_name="Pneumonia"
    )

    # ========================================================
    # TRAIN TB MODEL
    # ========================================================

    tb_train, tb_val, tb_test = (
        get_tb_datasets()
    )

    train_model(
        train_dataset=tb_train,
        val_dataset=tb_val,
        model_path=TB_MODEL_PATH,
        model_name="Tuberculosis"
    )

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)