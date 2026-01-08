YOLOv5 Video Analyzer

Prosta aplikacja desktopowa w Pythonie do analizy plików wideo z wykorzystaniem modelu YOLOv5, przeznaczona do wykrywania graczy w grze Counter-Strike 2 (CS2). Program rysuje bounding boxy na wykrytych obiektach i zapisuje wideo wynikowe.

Projekt wykonany w ramach studiów.

Funkcjonalności:
-analiza plików wideo (mp4, avi, mkv, mov)
-detekcja obiektów przy użyciu YOLOv5
-obsługa GPU (CUDA) oraz CPU
-graficzny interfejs użytkownika (Tkinter)
-zapis wideo wyjściowego z naniesionymi detekcjami

Technologie:
-Python 3.8+
-PyTorch
-YOLOv5
-OpenCV
-Tkinter
-NumPy

Uruchomienie:
- Zainstaluj wymagane biblioteki:
pip install torch torchvision opencv-python numpy
- Dołączycz niezbędny moduł yolov5 oraz plik modelu.
- Uruchom aplikację:
python main.py

Uwagi:
- model i rozmiar wejścia są ustawione na stałe
- aplikacja domyślnie próbuje użyć GPU, jeśli jest dostępne
- projekt ma charakter edukacyjny
