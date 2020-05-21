import torch
import sys
sys.path.insert(1, '/home/prajin/Desktop/balot_final/fuse_training_balot_classification/ml/src/data')

from loading_data import loading_datas
train_data, train_loader, validation_data, validation_loader, test_data, test_loader = loading_datas()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

model = load_checkpoint('checkpoint_best.pth')

def get_image(loader):
  images, labels = next(iter(loader))
  index = np.random.randint(len(labels))
  return images[index], labels[index], images, labels, index

def show_image(image, label):    
  img = image
  npimg = img.numpy()
  npimg = np.transpose(npimg, (1,2,0))
  plt.figure(figsize = (12,10))

  plt.imshow(npimg)
  print(label)

image, label,images, labels, index = get_image(test_loader)
show_image(image,label)

def predict(images, labels, index):
  indices, predicted = torch.max(model_c(images), 1)
  print("Predicted Class:", predicted[index].item())
  print("Actual Class:", labels[index].item())

predict(images, labels, index)