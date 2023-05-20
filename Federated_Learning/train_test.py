import torch
from timeit import default_timer as timer
from tqdm.auto import tqdm

# Device Agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,
               accuracy_fn, device: torch.device = device):
  
  train_loss, train_acc = 0, 0

  model.train()

  for batch, (x_train, y_train) in enumerate(dataloader):

    if device == 'cuda':
      x_train, y_train = x_train.to(device), y_train.to(device)
    

    # 1. Forward step
    pred = model(x_train)
    

    # 2. Loss
    # print(pred.shape)
    # print(y_train.shape)
    loss = loss_fn(pred, y_train)

    # 3. Grad zerostep
    optimizer.zero_grad()

    # 4. Backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # acc = accuracy_fn(y_train.squeeze(), torch.argmax(pred, dim=1))
    acc = accuracy_fn(torch.argmax(pred, dim=1), y_train)
    train_loss += loss
    train_acc += acc

  train_loss /= len(dataloader)
  train_acc /= len(dataloader)

  # print(train_loss, train_acc)
  return train_loss, train_acc, model


# test
def test_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module, accuracy_fn, 
              device: torch.device = device):
  
  test_loss, test_acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for x_test, y_test in dataloader:
      
      if device == 'cuda':
        x_test, y_test = x_test.to(device), y_test.to(device)

      # 1. Forward
      test_pred = model(x_test)
      
      # 2. Loss and accuray
      test_loss += loss_fn(test_pred, y_test)
      test_acc += accuracy_fn(y_test, torch.argmax(test_pred, dim=1))


    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

  # print(train_loss, train_acc)
  return test_loss, test_acc



def eval_func(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module, accuracy_fn, 
              device: torch.device = device):
  
  eval_loss, eval_acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for x_eval, y_eval in dataloader:
      
      if device == 'cuda':
        x_eval, y_eval = x_eval.to(device), y_eval.to(device)

      # 1. Forward
      eval_pred = model(x_eval)
      
      # 2. Loss and accuray
      eval_loss += loss_fn(eval_pred, y_eval)
      eval_acc += accuracy_fn(y_eval, torch.argmax(eval_pred, dim=1))


    eval_loss /= len(dataloader)
    eval_acc /= len(dataloader)

  # print(eval_loss, eval_acc)
  return eval_loss, eval_acc

# train client
def train_client(client_model, train_dataloader, test_dataloader, client_optimizer, client_loss_fn, accuracy_fn, device = device):
  # init. epochs
  epoches = 10

  client_model_train_loss, client_model_test_loss = [], []
  client_model_train_accs, client_model_test_accs = [], []
  print()

  start_time = timer()
  torch.manual_seed(64)
  torch.cuda.manual_seed(64)
  for epoch in tqdm(range(epoches)):
    # print(f"Epoch: {epoch+1}")
    train_loss, train_acc, client_model = train_loop(model = client_model, dataloader = train_dataloader,
                                      loss_fn = client_loss_fn, optimizer = client_optimizer,
                                      accuracy_fn = accuracy_fn, device = device)
    
    test_loss, test_acc = test_loop(model = client_model, dataloader = test_dataloader,
                                    loss_fn = client_loss_fn, accuracy_fn = accuracy_fn,
                                    device = device)
    
    client_model_train_loss.append(train_loss.item())
    client_model_test_loss.append(test_loss.item())
    client_model_train_accs.append(train_acc.item())
    client_model_test_accs.append(test_acc.item())


    # print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Accuray: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")
    # print()

  # plot_graph(model_18_train_loss, model_18_test_loss, model_18_train_accs, model_18_test_accs)

  end_time = timer()

  return client_model, client_optimizer, client_loss_fn, test_acc
  # print(f"Execution time: {end_time - start_time} Seconds.")

  # save_model('cifair100_model_18.pth', model_18)
  # print("Model saved")