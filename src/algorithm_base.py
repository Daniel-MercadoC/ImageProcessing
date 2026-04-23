import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed


def binarizacion_base(gray: np.ndarray) -> np.ndarray:
    """
    Realiza una binarizacion basica usando el metodo de Otsu.

    Parametros
    ----------
    gray : np.ndarray
        Imagen en escala de grises.

    Retorna
    -------
    np.ndarray
        Imagen binaria.

    Nota para el estudiante
    -----------------------
    Esta es una version base. El objetivo de la practica es que agregues
    un bloque de preprocesamiento antes de este paso para mejorar la entrada.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def segmentacion(binary: np.ndarray, min_distance: int = 10) -> np.ndarray:
    """
    Segmenta regiones usando transformada de distancia, maximos locales y watershed.
    """
    mask = binary > 0
    dist = ndi.distance_transform_edt(mask)

    coords = peak_local_max(dist, min_distance=min_distance, labels=mask)

    markers = np.zeros(dist.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    markers, _ = ndi.label(markers > 0)
    labels = watershed(-dist, markers, mask=mask)
    return labels


def circularidad(area: float, perimetro: float) -> float:
    """Calcula la circularidad de una region."""
    if perimetro == 0:
        return 0.0
    return 4 * np.pi * area / (perimetro ** 2)


def analizar(labels: np.ndarray) -> None:
    """Imprime metricas basicas para cada region segmentada."""
    props = regionprops(labels)
    for r in props:
        area = r.area
        circ = circularidad(r.area, r.perimeter)
        print(
            f"Region: {r.label:3d} | Area: {area:6.1f} | Circularidad: {circ:.4f}"
        )


def mostrar_imagenes(image, gray, imagen_para_segmentar, binary) -> None:
    """Muestra imagen original, gris, entrada a segmentacion y binarizacion."""
    cv2.imshow("1 - Imagen original", image)
    cv2.imshow("2 - Escala de grises", gray)
    cv2.imshow("3 - Imagen usada para segmentar", imagen_para_segmentar)
    cv2.imshow("4 - Binarizacion", binary)
    print("\nPresiona cualquier tecla sobre una ventana para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ejecutar(image_path: str) -> None:
    """
    Flujo principal:
    1. Cargar imagen
    2. Convertir a gris
    3. Aplicar preprocesamiento (bloque para el estudiante)
    4. Binarizar
    5. Segmentar
    6. Analizar regiones
    7. Mostrar resultados intermedios
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ============================================================
    # BLOQUE DE PREPROCESAMIENTO
    # ============================================================
    # Aqui debes agregar tu propuesta de preprocesamiento.
    # Algunas opciones que puedes probar:
    # o filtros para reducir ruido
    # o ecualizacion de histograma
    # o CLAHE
    # o operaciones morfologicas
    # - correccion de iluminacion
    #
    # Version base: sin preprocesamiento.

    imagen_para_segmentar = gray.copy()
    denoising = cv2.fastNlMeansDenoising(gray)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(denoising, cv2.MORPH_OPEN, kernel, iterations = 2)
    # erosion = cv2.morphologyEx(gray, cv2.MORPH_ERODE, kernel, iterations = 2)
    # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # imagen_para_segmentar = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    
    # imagen_hist_eq = gray.copy()
    # cv2.equalizeHist(gray, imagen_hist_eq)
    # ------------------------------------------------------------
    # EJEMPLO BASICO COMENTADO (referencia, NO activo)
    # ------------------------------------------------------------
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # imagen_clahe = clahe.apply(gray)
    # imagen_filtrada = cv2.medianBlur(imagen_clahe, 7)
    # imagen_para_segmentar = imagen_filtrada
    # ------------------------------------------------------------

    imagen_para_segmentar = opening
    
    binary = binarizacion_base(imagen_para_segmentar)
    labels = segmentacion(binary)
    analizar(labels)

    total = len(regionprops(labels))
    print(f"\nTotal de regiones detectadas: {total}")

    mostrar_imagenes(image, gray, imagen_para_segmentar, binary)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python algorithm_base.py <ruta_imagen>")
        sys.exit(1)
    ejecutar(sys.argv[1])
