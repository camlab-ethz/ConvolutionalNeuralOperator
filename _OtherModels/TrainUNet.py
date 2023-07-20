import copy
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from Problems.BenchMarksUNet import NavierStokes_VIDON, ShearLayer, EquationModel

if len(sys.argv) == 1:

    training_properties = {
        "learning_rate": 0.0005,
        "weight_decay": 1e-12,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "epochs": 1000,
        "batch_size": 20,
        "exp": 1,
        "training_samples": 500,
    }
    model_architecture_ = {
        "FourierF": 2,
        "retrain": 4,
        "channels": 8
    }
    # which_example = "darcy"
    #which_example = "navier_stokes_vidon"
    # which_example = "shear_layer_rec_out"
    # which_example = "advection"
    which_example = "airfoil"
    # which_example = "shear_layer64"
    # which_example = "shear_layer_rec"

    folder = "TrainedModels/UNEtTest"

else:
    folder = sys.argv[1]
    training_properties = json.loads(sys.argv[2].replace("\'", "\""))
    model_architecture_ = json.loads(sys.argv[3].replace("\'", "\""))
    which_example = sys.argv[4]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
writer = SummaryWriter(log_dir=folder)

print("#####################################################")
print("Training UNet on ", device)
print("#####################################################")

learning_rate = training_properties["learning_rate"]
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
weight_decay = training_properties["weight_decay"]
scheduler_step = training_properties["scheduler_step"]
scheduler_gamma = training_properties["scheduler_gamma"]
p = training_properties["exp"]
training_samples = training_properties["training_samples"]

if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([model_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')


if which_example == "navier_stokes_vidon":
    example = NavierStokes_VIDON(model_architecture_, device, batch_size)
elif which_example == "shear_layer":
    example = ShearLayer(model_architecture_, device, batch_size, 750)
elif which_example == "poisson":
    example = EquationModel(model_architecture_, device, batch_size, training_samples=1024, which_data="poisson")
elif which_example == "wave":
    example = EquationModel(model_architecture_, device, batch_size, training_samples=512, which_data="wave")
elif which_example == "allen_cahn":
    example = EquationModel(model_architecture_, device, batch_size, training_samples=256, which_data="allen_cahn")
elif which_example == "cont_t":
    example = EquationModel(model_architecture_, device, batch_size, training_samples=512, which_data="cont_t")
elif which_example == "discont_t":
    example = EquationModel(model_architecture_, device, batch_size, training_samples=512, which_data="discont_t")
elif which_example == "airfoil":
    example = EquationModel(model_architecture_, device, batch_size, training_samples=512, which_data="airfoil")

else:
    raise ValueError()

model = example.model

train_loader = example.train_loader
test_loader = example.test_loader


n_params = model.print_size()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
freq_print = 1
if p == 1:
    loss = torch.nn.L1Loss()
elif p == 2:
    loss = torch.nn.MSELoss()
best_model_testing_error = 100
patience = int(0.1 * epochs)
counter = 0
for epoch in range(epochs):
    with tqdm(unit="batch", disable=False) as tepoch:
        model.train()
        tepoch.set_description(f"Epoch {epoch}")
        train_mse = 0.0
        running_relative_train_mse = 0.0
        for step, (input_batch, output_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)

            output_pred_batch = model(input_batch)

            if which_example == "airfoil":
                output_pred_batch[input_batch == 1] = 1
                output_batch[input_batch == 1] = 1

            loss_f = loss(output_pred_batch, output_batch) / loss(torch.zeros_like(output_batch).to(device), output_batch)

            loss_f.backward()
            optimizer.step()
            train_mse = train_mse * step / (step + 1) + loss_f.item() / (step + 1)
            tepoch.set_postfix({'Batch': step + 1, 'Train loss (in progress)': train_mse})

        writer.add_scalar("train_loss/train_loss", train_mse, epoch)

        with torch.no_grad():
            model.eval()
            test_relative_l2 = 0.0
            for step, (input_batch, output_batch) in enumerate(test_loader):
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                output_pred_batch = model(input_batch)

                if which_example == "airfoil":
                    output_pred_batch[input_batch == 1] = 1
                    output_batch[input_batch == 1] = 1

                loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(test_loader)

            writer.add_scalar("val_loss/val_loss", test_relative_l2, epoch)

            if test_relative_l2 < best_model_testing_error:
                best_model_testing_error = test_relative_l2
                best_model = copy.deepcopy(model)
                torch.save(best_model, folder + "/model.pkl")
                writer.add_scalar("val_loss/Best Relative Testing Error", best_model_testing_error, epoch)
                counter = 0
            else:
                counter += 1

        tepoch.set_postfix({'Train loss': train_mse, "Relative Val loss": test_relative_l2})
        tepoch.close()

        with open(folder + '/errors.txt', 'w') as file:
            file.write("Training Error: " + str(train_mse) + "\n")
            file.write("Best Testing Error: " + str(best_model_testing_error) + "\n")
            file.write("Current Epoch: " + str(epoch) + "\n")
            file.write("Params: " + str(n_params) + "\n")
        scheduler.step()

        if counter > patience:
            print("Early Stopping")
            break
