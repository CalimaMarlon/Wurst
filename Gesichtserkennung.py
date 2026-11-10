import cv2
from picamera2 import Picamera2

# Kamera starten
kamera = Picamera2()

config = kamera.create_preview_configuration(
    main={"size": (640, 480)}
)

kamera.configure(config)
kamera.start()

# Gesichtserkennung laden
gesicht = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Lächelerkennung laden
laecheln = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_smile.xml"
)

print("KI-Gesichtserkennung gestartet!")

while True:

    # Bild aufnehmen
    bild = kamera.capture_array()

    # RGB zu Graustufen
    grau = cv2.cvtColor(
        bild,
        cv2.COLOR_RGB2GRAY
    )

    # Gesichter suchen
    gesichter = gesicht.detectMultiScale(
        grau,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in gesichter:

        # Gesicht markieren
        cv2.rectangle(
            bild,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Gesicht ausschneiden
        gesicht_bild = grau[y:y+h, x:x+w]

        # Lächeln suchen
        smile = laecheln.detectMultiScale(
            gesicht_bild,
            scaleFactor=1.7,
            minNeighbors=20
        )

        if len(smile) > 0:

            emotion = "GLUECKLICH"
            farbe = (0, 255, 0)

        else:

            emotion = "KEIN LAECHELN"
            farbe = (0, 0, 255)

        # Emotion anzeigen
        cv2.putText(
            bild,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            farbe,
            2
        )

    # Bild anzeigen
    cv2.imshow(
        "Meine KI Gesichtserkennung",
        cv2.cvtColor(bild, cv2.COLOR_RGB2BGR)
    )

    # Mit ESC beenden
    if cv2.waitKey(1) & 0xFF == 27:
        break

kamera.stop()
cv2.destroyAllWindows()