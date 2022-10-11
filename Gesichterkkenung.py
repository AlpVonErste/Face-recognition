import cv2

import inline as inline
import matplotlib

import matplotlib.pyplot as plt # Matplotlib wurde eigenlich eintwickelt, um Grafiken zu bearbeiten.
import numpy as np # Mit Nupmy ist das effiziente Rechnen mit Matrizen, mehrdimensionalen Arrays und Vektoren möglich

img = cv2.imread("Alperen_Tunc.jpeg") #Wählen die Datei.

img.shape # Diese Befehl hilft uns, zu sehen, wie viele Zeilen und Spalten vorhanden ist.

plt.imshow(img) # um Bild mithilfe Matplot zu zeigen.
#plt.show() # Bilder sind in RGB Format , aber wir versuchen gerade, in BGR(cv2) zeigen zu lassen,
                #Deswegen sind Farben Anderes dargestellt.

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Konvertiere ich jtzt BGR zu RGA, um Bilder Richtig zu sehen.

plt.imshow(img_rgb)
#plt.show()

print(img[0][0])
print(img_rgb[0][0]) # Kontrolieren von Umwandlung

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Konwertiere ich Daten zu Gray, um schneller berarbeiten zu können.

print(img.shape)
print(img_gray.shape) # Ein Diemension weniger

plt.imshow(img_gray, "gray") # Ursprünglich Matplotlib ist für Mathematiche Berechnungen und zeigt immer die Werte Gelb,
                                #deswegen schreibe ich genau in Welchem Frabe ich das Bild sehen will.
#plt.show()


classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml") # Tranierte GesichtDatei

faces = classifier.detectMultiScale(img_gray) # um unterschidliche größe Köpfe zu finden

print(faces) # Drei Gesichte wurde Erkannt

c = img.copy() #Eine Koppie erstellen
for face in faces: # in Diesem Koppie Gesicht zu Gesicht Bestimmen
    x, y, w, h = face # x und y Achse sowie Bereite und Höhe
    cv2.rectangle(c, (x, y), (x + w, y + h), (0, 0, 255), 10)
    # Wie Groß Kästchen sein soll und welsche Farbe haben soll sowie Bereite von Rahmen Bestimme ich
    print(face)

img_rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
#plt.show()

# behebe ich Fehler, weil computer 3 Gesichte erkannt hat, indem ich Rote Kästchen schieben.
faces = classifier.detectMultiScale(img_gray, minNeighbors=10)

c = img.copy() #Eine Koppie erstellen
for face in faces: # in Diesem Koppie Gesicht zu Gesicht Bestimmen
    x, y, w, h = face # x und y Achse sowie Bereite und Höhe
    cv2.rectangle(c, (x, y), (x + w, y + h), (0, 0, 255), 20)
    # Wie Groß Kästchen sein soll und welsche Farbe haben soll sowie Bereite von Rahmen Bestimme ich
    print(face)

img_rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()




