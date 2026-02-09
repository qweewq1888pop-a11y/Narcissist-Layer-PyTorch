# Narcissist-Layer-PyTorch
Narcissist-Layer-PyTorch
Thanks To cifar-10 The model was benchmarked using the CIFAR-10 dataset collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
the result:
    PS C:\Yeni klasör (16)> & C:/Users/aydin/AppData/Local/Programs/Python/Python311/python.exe "c:/Yeni klasör (16)/Aydintraib.py"
🔥 Çalışma Ortamı: cuda
📦 Veriler indiriliyor...
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data\cifar-10-python.tar.gz
100%|████████████████| 170M/170M [00:35<00:00, 4.77MB/s]
Extracting ./data\cifar-10-python.tar.gz to ./data
C:\Users\aydin\AppData\Local\Programs\Python\Python311\Lib\site-packages\torchvision\datasets\cifar.py:83: VisibleDeprecationWarning: dtype(): align should be passed as Python or NumPy boolean but got `align=0`. Did you mean to pass a tuple to create a subarray type? (Deprecated NumPy 2.4)
  entry = pickle.load(f, encoding="latin1")
Files already downloaded and verified

🚀 Eğitim Başlıyor (Narsist Mod)...
[1,   200] loss: 1.725 | acc: 38.05%
[1,   400] loss: 1.409 | acc: 43.90%
[1,   600] loss: 1.296 | acc: 47.61%
[2,   200] loss: 1.145 | acc: 60.35%
[2,   400] loss: 1.099 | acc: 61.30%
[2,   600] loss: 1.046 | acc: 62.24%
[3,   200] loss: 0.943 | acc: 67.73%
[3,   400] loss: 0.935 | acc: 67.67%
[3,   600] loss: 0.937 | acc: 67.75%
✅ Eğitim Bitti!

🔍 Narsist Katman Analizi:
features.0.weight: Ortalama Değer = -0.0059
features.3.weight: Ortalama Değer = 1.1679
features.4.weight: Ortalama Değer = -0.0124
features.7.weight: Ortalama Değer = 1.2190
PS C:\Yeni klasör (16)>  this is 60.458 parameter model 
