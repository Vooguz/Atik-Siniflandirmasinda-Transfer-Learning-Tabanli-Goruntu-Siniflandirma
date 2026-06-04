"""README icin degerlendirme gorselleri uretir.
Egitilmis modelleri (models/) yukler, validation seti uzerinde
degerlendirip assets/ klasorune PNG gorseller kaydeder.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre
from sklearn.metrics import classification_report, confusion_matrix

os.makedirs("assets", exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
dataset_path = "data/original"

print(">> Validation seti yukleniyor...")
val_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path, validation_split=0.2, subset="validation",
    seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
)
class_names = val_dataset.class_names
num_classes = len(class_names)
val_ds = val_dataset.cache().prefetch(tf.data.AUTOTUNE)
print("Siniflar:", class_names)

print(">> Modeller yukleniyor...")
resnet_model = tf.keras.models.load_model(
    "models/resnet50_finetuned.keras", safe_mode=False,
    custom_objects={"preprocess_input": res_pre})
mobilenet_model = tf.keras.models.load_model(
    "models/mobilenetv2.keras", safe_mode=False,
    custom_objects={"preprocess_input": mob_pre})
scratch_model = tf.keras.models.load_model(
    "models/scratch_cnn.keras", safe_mode=False)

print(">> Degerlendirme (evaluate)...")
res_loss, res_acc = resnet_model.evaluate(val_ds, verbose=0)
mob_loss, mob_acc = mobilenet_model.evaluate(val_ds, verbose=0)
scr_loss, scr_acc = scratch_model.evaluate(val_ds, verbose=0)
print(f"ResNet50    acc={res_acc:.4f} loss={res_loss:.4f}")
print(f"MobileNetV2 acc={mob_acc:.4f} loss={mob_loss:.4f}")
print(f"Scratch CNN acc={scr_acc:.4f} loss={scr_loss:.4f}")

# --- 1) Accuracy karsilastirma bar chart ---
models = ["ResNet50", "MobileNetV2", "Scratch CNN"]
accs = [res_acc, mob_acc, scr_acc]
colors = ["#2563eb", "#16a34a", "#dc2626"]
plt.figure(figsize=(7, 5))
bars = plt.bar(models, accs, color=colors)
for b, a in zip(bars, accs):
    plt.text(b.get_x() + b.get_width()/2, a + 0.01, f"{a:.3f}",
             ha="center", va="bottom", fontweight="bold")
plt.title("Model Validation Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("assets/accuracy_comparison.png", dpi=130)
plt.close()
print(">> assets/accuracy_comparison.png kaydedildi")

# --- En iyi model ---
best_acc, best_model = max(
    [(res_acc, resnet_model), (mob_acc, mobilenet_model), (scr_acc, scratch_model)],
    key=lambda t: t[0])
best_name = {id(resnet_model): "ResNet50", id(mobilenet_model): "MobileNetV2",
             id(scratch_model): "Scratch CNN"}[id(best_model)]
print(f">> En iyi model: {best_name} (acc={best_acc:.4f})")

print(">> Tahminler toplaniyor (confusion matrix icin)...")
y_true, y_pred = [], []
for images, labels in val_ds:
    preds = best_model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

report = classification_report(y_true, y_pred, target_names=class_names)
with open("assets/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(f"En iyi model: {best_name} (val accuracy = {best_acc:.4f})\n\n")
    f.write(report)
print(report)

# --- 2) Confusion matrix ---
cmtx = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 7))
plt.imshow(cmtx, cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.colorbar()
plt.xticks(range(num_classes), class_names, rotation=90)
plt.yticks(range(num_classes), class_names)
thresh = cmtx.max() / 2.0
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, cmtx[i, j], ha="center", va="center",
                 color="white" if cmtx[i, j] > thresh else "black", fontsize=8)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("assets/confusion_matrix.png", dpi=130)
plt.close()
print(">> assets/confusion_matrix.png kaydedildi")

# --- 3) Grad-CAM (ResNet50) ---
try:
    print(">> Grad-CAM uretiliyor...")
    last_conv = "conv5_block3_out"
    resnet_base = None
    for layer in resnet_model.layers:
        if "resnet50" in layer.name.lower() or isinstance(layer, tf.keras.Model):
            resnet_base = layer
            break
    conv_model = tf.keras.Model(resnet_base.input,
                                resnet_base.get_layer(last_conv).output)
    classifier_input = tf.keras.Input(shape=resnet_base.get_layer(last_conv).output.shape[1:])
    x = classifier_input
    for layer in resnet_model.layers[3:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    def gradcam(img_array):
        pre = res_pre(tf.identity(img_array))
        with tf.GradientTape() as tape:
            conv_out = conv_model(pre)
            tape.watch(conv_out)
            preds = classifier_model(conv_out)
            idx = tf.argmax(preds[0])
            channel = preds[:, idx]
        grads = tape.gradient(channel, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        hm = conv_out @ pooled[..., tf.newaxis]
        hm = tf.squeeze(hm)
        hm = tf.maximum(hm, 0) / (tf.math.reduce_max(hm) + 1e-8)
        return hm.numpy(), int(idx)

    # 3 ornek goruntu icin Grad-CAM
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for image_batch, _ in val_ds.take(1):
        for k in range(3):
            image = image_batch[k]
            hm, idx = gradcam(tf.expand_dims(image, 0))
            img = image.numpy().astype("uint8")
            hm8 = np.uint8(255 * hm)
            jet = cm.get_cmap("jet")(np.arange(256))[:, :3]
            jet_hm = jet[hm8]
            jet_hm = tf.keras.utils.array_to_img(jet_hm).resize((img.shape[1], img.shape[0]))
            jet_hm = tf.keras.utils.img_to_array(jet_hm)
            superimposed = np.clip(jet_hm * 0.4 + img, 0, 255).astype("uint8")
            axes[0, k].imshow(img); axes[0, k].set_title("Original"); axes[0, k].axis("off")
            axes[1, k].imshow(superimposed)
            axes[1, k].set_title(f"Grad-CAM: {class_names[idx]}"); axes[1, k].axis("off")
        break
    plt.suptitle("Grad-CAM (ResNet50)", fontsize=14)
    plt.tight_layout()
    plt.savefig("assets/gradcam.png", dpi=130)
    plt.close()
    print(">> assets/gradcam.png kaydedildi")
except Exception as e:
    print(">> Grad-CAM atlandi:", repr(e))

print(">> BITTI")
