from simple_autoencoders import *


# class AutoEncoder:
#     def __init__(self, filename):
#         self.model = NeuralNetwork()
#         self.model.load_state_dict(torch.load(filename))
    
#     def decode(self, latent_space_vector):
#         result = torch.matmul(self.model.theta3, latent_space_vector)
#         result = activation_function(result)
#         result = torch.matmul(self.model.theta4, result)
#         result = activation_function(result)
#         result = torch.matmul(self.model.theta_final, result)
#         return result
    
#     def encode(self, image):
#         result = torch.matmul(self.model.theta1, image.t())
#         result = activation_function(result)
#         result = torch.matmul(self.model.theta2, result)
#         result = activation_function(result)

#         result = torch.matmul(self.model.choke, result)
#         result = activation_function(result)

#         return result


class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True))

        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,1,kernel_size=5),
            nn.ReLU(True),
            nn.Sigmoid())

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    # load the model and pass it into encoder
    auto = Autoencoder() 
    auto.load_state_dict(torch.load("rando.pt"))
    auto.eval()

    # look at the first image in the dataset, a 5
    test_image, label = train_set[2]
    print(f"Label: {label}")
    show(test_image)

    # encode it
    encoding = auto.encoder(test_image.reshape(1,28,28))
    print(f"Encoding: {encoding}")

    # now decode it
    out = auto.decoder(encoding)
    out = out.detach().numpy()
    out += 10
    show(out)
    input()


