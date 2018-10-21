# Hinweise

Dieses Programm wurde in spyder entwickelt, getestet und ausgeführt (https://github.com/spyder-ide/spyder).
Die verwendete Python-Version ist 3.5.2.
Dieses Programm verwendet folgende Python-Bibliotheken:

- numpy
- scipy
- scikit-learn
- pillow
- h5py
- keras
- pydot
- Augmentor
- pandas
- seaborn
- tensorflow-gpu (für GPU-Unterstützung)

Alle Python-Bibliotheken wurden selbst compiliert. Dies war insbesondere für tensorflow-gpu nötig, dessen binary-Version keine GPU-Unterstützung für NVIDIA-Grafikkarten mit CUDA-Kompatibilität kleiner als 3.5 anbietet. Die meisten Python-Bibliotheken lassen sich mittels pip aus dem Quellcode mittels folgendem Befehl compilieren und installieren:

pip install \<name\> --no-binary :all:
  
Eine Anleitung zum compilieren von tensorflow findet sich hier: https://www.tensorflow.org/install/source

Für GPU-Unterstützung sind zudem weitere Schritte vonnöten. Insbesondere die Installation des CUDA-Toolkits, NVIDIA GPU-Treiber und weiterer Software. Eine ausführliche Anleitung findet sich hier: https://www.tensorflow.org/install/gpu

## Test auf GPU-Unterstützung

Um zu testen, ob die GPU des Systems erfolgreich von tensorflow verwendet wird, sollte folgender Code in einer Python-Shell ausgeführt werden:

```
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```

In der Ausgabe der Konsole sollte folgender Eintrag auftauchen (in diesem Falle für eine GeForce GTX 680):
```
2018-10-22 00:28:19.398747: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:964] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-10-22 00:28:19.399243: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1411] Found device 0 with properties: 
name: GeForce GTX 680 major: 3 minor: 0 memoryClockRate(GHz): 1.2015
pciBusID: 0000:01:00.0
totalMemory: 1.95GiB freeMemory: 1.61GiB
2018-10-22 00:28:19.399271: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1490] Adding visible gpu devices: 0
2018-10-22 00:28:19.655303: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-10-22 00:28:19.655334: I tensorflow/core/common_runtime/gpu/gpu_device.cc:977]      0 
2018-10-22 00:28:19.655341: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990] 0:   N 
2018-10-22 00:28:19.655489: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1103] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1381 MB memory) -> physical GPU (device: 0, name: GeForce GTX 680, pci bus id: 0000:01:00.0, compute capability: 3.0)
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 680, pci bus id: 0000:01:00.0, compute capability: 3.0
2018-10-22 00:28:19.672038: I tensorflow/core/common_runtime/direct_session.cc:291] Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 680, pci bus id: 0000:01:00.0, compute capability: 3.0
```
Tensorflow verwendet automatisch die CPU, sollte GPU Unterstützung nicht möglich sein.

# BridgeTrainer.py
Traininert ein neuronales Netz mithilfe von Bilddaten und führt Klassifizierungen durch.

## Command Line Arguments
| Kurzform | Langform         | Funktion                              |
| -------- | ---------------- | ------------------------------------- |
| -p       | --imagepath=     | Pfad zum Verzeichnis mit Bilddaten.                                                     |
| -d       | --modelDir=      | (Optional) Pfad zum Verzeichnis, in welches das Modell, traininerte Gewichte und andere Daten gespeichert werden. Wenn nicht angegeben, wird ein Name aus den Parametern des Programms kreiert.                       |
| -w       | --imageWidth=    | (Optional) Default=30. Gibt die Breite in Pixel an, in die alle Bilddaten im Bildverzeichnis zum Zweck des Trainings, Validierens und Testens skaliert werden.                                                                     |
| -h       | --imageHeight=   | (Optional) Default=30. Gibt die Höhe in Pixel an, in die alle Bilddaten im Bildverzeichnis zum Zweck des Trainings, Validierens und Testens skaliert werden.                                                                     |
| -e       | --epochs=        | (Optional) Default=20. Gibt die Anzahl an Trainingsepochen für das neuronale Netz an.   |
| -b       | --batchSize=     | (Optional) Default=8. Gibt die Batchgröße an, die während des Trainings verwendet wird. |
| -t       | --retrain        | Wenn angegeben, bewirkt, ob bereits gespeicherte Gewichtungen gelöscht und das neuronale Netz im angegebenem Modellverzeichnis erneut trainiert werden soll. Hat keine Auswirkungen, wenn keine Gewichtungen im Modellverzeichnis existieren. |
| -a       | --architecture=  | (Optional) Default=cnn. Gibt an, welche Netzwerkarchitektur zum Training verwendet werden soll. Folgende Werte stehen zur Auswahl zur Verfügung: AlexNet, cnn_extended, cnn, reduced_cnn, reduced_cnn2, dense, reduced_dense, 1l_dense                                                                                                                |
| -l       | --learning-rate= | (Optional) Default=0.01. Lernrate des neuronalen Netzwerkes.                            |
| -o       | --optimizer=     | (Optional) Default=sgd. Verwendeter Algorithmus. Mögliche Werte: sgd, adam, rmsprop     |


Wenn das Programm ohne das --retrain-Argument ausgeführt wird und Gewichtungen für ein Modell bereits existieren, wird das Modell nicht erneut trainiert, und nur ein Klassifizierungsdurchgang durchgeführt.

Wenn nach 50 Epochen keine Verbesserung in der Genauigkeit des Netzwerkes festgestellt werden kann, bricht das Programm automatisch weiteres Training ab.


## Modellverzeichnis
Sofern kein Modellverzeichnis angegeben, kreiert das Programm aus folgenden Parametern selbstständig einen Ordnernamen:

- Architektur {a]
- Bildbreide {w}
- Bildhöhe {h}
- Epochen {e}
- Batchgröße {b}
- Lernrate {lr}
- Optimizer {o}
- Bilddaten-Verzeichnis {dir}

Der Verzeichnisname folgt folgendem Format:
{a}\_{w}x{h}\_e{e}\_b{b}\_lr{lr}\_o{o}\_{dir}

**Beispiel:** cnn_30x30_e20_b8_lr0.01_osgd_bridges

### Inhalt
In das Modellverzeichnis werden folgende Dateien während des Trainings und Testens des neuronalen Netzes gespeichert.

| Datei | Bedeutung |
| ----- | --------- |
| model.json | Beinhaltet die Netzwerkarchitektur im json-Format. |
| weights.h5 | Beinhaltet die Gewichtungen des Modells. |
| model.svg | Eine grafische Abbildung des Netzmodells. |
| acc.png | Ein Graph, welcher die Genauigkeit des Modells während des Trainings darstellt. |
| loss.png | Ein Graph, welcher die Fehlerrate des Modells während des Trainings darstellt. |
| val_acc.png | Ein Graph, welcher die Genauigkeit des Modells während des Trainings darstellt, in dem es gegen ein Validation-Set geprüft wurde, mit dem es keine Schnittmenge mit dem Trainingsset gibt. |
| val_loss.png | Ein Graph, welcher die Fehlerrate des Modells während des Trainings darstellt, in dem es gegen ein Validation-Set geprüft wurde, mit dem es keine Schnittmenge mit dem Trainingsset gibt. |
| conf_matrix.png | Eine Konfusionsmatrix, die angibt, wie genau das Netz Bilder den einzelnen Klassen zuordnet. **Wichtig!** Momentan wird aufgrund einer Einschränkung in der dafür verwendeten Bibliothek dieses Bild nicht automatisch angelegt, sondern muss manuell nach einem Klassifizierungsdurchgang aus der Spyder-Konsole kopiert werden. |
| notes.txt | Beinhaltet Informationen zu Trainings- und Klassifizierungsdurchgängen wie Trainingsdauer, verwendete Parameter, Anzahl Bilder, Anzahl Labels, Anzahl Klassen, etc. |
| predictions.png | Eine Grafik, die alle Bilder aus dem Test-Datenset mit ihrem dazugehörigem tatsächlichem, und vom Netz zugewiesenem Label darstellt. | 
| predictions.txt | Eine Textform von predictions.png. |

## Anforderungen an das Bildverzeichnis
Das Verzeichnis, dass die Bilddaten beinhaltet, muss folgende Anforderungen erfüllen:

- Das Verzeichnis muss alle Bilder die zum Trainieren, Validieren und Testen verwendet werden, beinhalten.
- Das Verzeichnis beinhaltet eine Datei namens train.csv, welches die Namen der Bilder und ihr dazugehöriges Label beinhaltet, die zum Training verwendet werden soll.
- Das Verzeichnis beinhaltet eine Datei namens test.csv, welches die Namen der Bilder und ihr dazugehöriges Label beinhaltet, die zum Testen verwendet werden soll.
- Die Bilddaten müssen im jpeg-Format abgespeichert sein und die .jpg Dateiendung beinhalten.

Einträge der train.csv und test.csv müssen folgendem Format folgen:

\<Dateiname ohne Endung\>:\<Label\>
  
**Beispiel**: 2452552:5

# confusion_matrix_pretty_print.py
Bibliothek zum Rendern von Konfusionsmatritzen für die Bilderklassifizierung.
