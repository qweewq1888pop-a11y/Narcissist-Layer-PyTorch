import torch  # <--- İŞTE BU EKSİKTİ
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from AydinCore import Narcissist  # Dosya adın AydinCore.py ise bu doğru

# --- Cihaz Seçimi (CUDA varsa affetme) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Çalışma Ortamı: {device}")

# --- 1. Veri Seti (CIFAR-10) ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("📦 Veriler indiriliyor...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# Windows için num_workers=0 yaptım garanti olsun diye, sorun çıkmazsa 2 yapabilirsin
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

# --- 2. Senin Narsist Modelin ---
class NarsistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Katman 1: Standart Conv
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Katman 2: SENİN NARSİST KATMANIN
            Narcissist(num_features=32, dim=(2,3)), 
            
            # Katman 3: Derinleşiyoruz
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Katman 4: Bir doz daha Narsistlik
            Narcissist(num_features=64, dim=(2,3)),
            
            nn.Flatten()
        )
        self.classifier = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Modeli oluştur
model = NarsistCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 3. Eğitim Döngüsü ---
print("\n🚀 Eğitim Başlıyor (Narsist Mod)...")

for epoch in range(3):  # Hızlı test için 3 Epoch
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Anlık Accuracy hesabı
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 200 == 199:    # Her 200 batch'te bir durum raporu
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f} | acc: {100 * correct / total:.2f}%")
            running_loss = 0.0

print("✅ Eğitim Bitti!")

# --- 4. Narsist Katmanların Durumu ---
print("\n🔍 Narsist Katman Analizi:")
for name, param in model.named_parameters():
    if "weight" in name and "features" in name and param.shape[0] < 100: 
        print(f"{name}: Ortalama Değer = {param.data.mean().item():.4f}")