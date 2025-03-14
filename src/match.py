import json
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Polygon
import argparse
from lxml import etree
import line_based_matching as lm
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.cluster import DBSCAN

# Function to extract rotation and scale from a transformation matrix
def decompose_matrix(T):
    # Assuming T is a 4x4 transformation matrix
    rotation = T[:3, :3]
    scale = np.linalg.norm(rotation, axis=0)  # Get the scale factors (diagonal of the matrix)
    rotation_normalized = rotation / scale  # Normalize rotation part
    translation = T[:3, 3]
    
    return rotation_normalized, scale, translation

# Function to calculate the Frobenius norm (rotation distance)
def frobenius_norm(A, B):
    return np.linalg.norm(A - B, 'fro')

# Function to calculate Euclidean distance (translation distance)
def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

# Function to calculate scale difference (using Euclidean norm)
def scale_distance(scale1, scale2):
    return np.linalg.norm(scale1 - scale2)

# Function to calculate the distance between two transformation matrices with scale
def transformation_distance(T1, T2, alpha=1.0, beta=1.0, gamma=1.0):
    # Decompose matrices into rotation, scale, and translation
    R1, scale1, t1 = decompose_matrix(np.array(T1))
    R2, scale2, t2 = decompose_matrix(np.array(T2))
    
    # Calculate individual distances
    d_rot = frobenius_norm(R1, R2)  # Rotation distance (Frobenius norm)
    d_trans = euclidean_distance(t1, t2)  # Translation distance (Euclidean distance)
    d_scale = scale_distance(scale1, scale2)  # Scale distance
    
    #print("dist",d_rot,d_trans,d_scale)
    # Total distance as weighted sum of individual distances
    d_total = alpha * d_rot + beta * d_trans + gamma * d_scale
    return d_total

#def transformation_distance(T1, T2):
#    d1 = abs(np.trace(np.dot(T1, np.linalg.inv(T2)))-4)
#    d2 = abs(np.trace(np.dot(T2, np.linalg.inv(T1)))-4)
#    return max(d1,d2)

def softmax(x):
    """Compute softmax probabilities for a list of values."""
    exp_x = np.exp(x - np.max(x))  # Subtract max to avoid numerical issues
    return exp_x / np.sum(exp_x)
  
def cluster_transforms_dbscan(labels, scores, transformations, eps=3.0, min_samples=1):
    """
    Cluster transformation matrices using DBSCAN and calculate sum scores & representative matrix.

    Parameters:
        transformations (list of np.ndarray): List of 4x4 transformation matrices.
        scores (list of float): Scores associated with each transformation.
        eps (float): Distance threshold for clustering.
        min_samples (int): Minimum number of points in a cluster.

    Returns:
        dict: Cluster information with sum score and representative matrix.
    """
    n = len(transformations)

    # Compute pairwise distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist_matrix[i, j] = transformation_distance(transformations[i], transformations[j], 12.0, 0.1,1.0)
            dist_matrix[j, i] = dist_matrix[i, j]  # Symmetric matrix
            #print(i,j, dist_matrix[i, j] )


    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    cluster_labels = dbscan.fit_predict(dist_matrix)

    # Organize clusters
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label == -1:
            continue  # Ignore noise points
        if label not in clusters:
            clusters[label] = {"matrices": [], "scores": [], "labels": []}
        clusters[label]["matrices"].append(transformations[i])
        clusters[label]["scores"].append(scores[i])
        clusters[label]["labels"].append(labels[i])

    # Compute sum score and representative matrix for each cluster
    cluster_results = []
    for label, data in clusters.items():
        sum_score = max(data["scores"])
        # Compute representative matrix (mean of transformation matrices)
        representative_matrix = np.mean(data["matrices"], axis=0)
        combined_labels = ", ".join(data["labels"])  # Combine all labels into a single string

        cluster_results.append({
            "score": sum_score,
            "matrix": representative_matrix.tolist(),
            "label":combined_labels
        })

    # Extract scores, apply softmax, and update back
    cluster_scores = np.array([item["score"] for item in cluster_results])
    softmax_scores = np.exp(cluster_scores - np.max(cluster_scores)) / np.sum(np.exp(cluster_scores - np.max(cluster_scores)))

    # Update the list with normalized scores
    for i, item in enumerate(cluster_results):
        item["score"] = softmax_scores[i]
    
    cluster_results.sort(key=lambda x: x["score"], reverse=True)
    return cluster_results

def get_utm_epsg(lat, lon):
    """Calculates the UTM EPSG code using a simple formula."""
    zone = int((lon + 180) / 6) + 1
    epsg_code = 32600 + zone if lat >= 0 else 32700 + zone
    return epsg_code

def read_kml(kml_file):
    tree = etree.parse(kml_file)
    root = tree.getroot()

    # KML namespace
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    # Find all Placemarks under Folder or Document
    placemarks = root.findall('.//kml:Placemark', ns)
    polygons = []
    
    for placemark in placemarks:
        polygon = placemark.find('.//kml:Polygon', ns)
        if polygon is not None:
            coordinate_string = polygon.find('.//kml:coordinates', ns).text.strip()
            coordinates = coordinate_string.strip().split(" ")
            coordinate_list = [tuple(map(float, coord.split(","))) for coord in coordinates]
            polygons.append(Polygon(coordinate_list))
    return polygons

def read_json(json_file):
    """Reads a JSON file and extracts the bounds as a polygon."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    bounds_wkt = data.get("bounds", "")
    clip_polygon = wkt.loads(bounds_wkt)  # Convert WKT string to a Shapely Polygon
    return clip_polygon

def clip_kml_polygons(kml_polygons, clip_polygon):
    """Clips KML polygons using the WKT polygon from JSON."""
    clipped_polygons = [poly.intersection(clip_polygon) for poly in kml_polygons]
    return [p for p in clipped_polygons if not p.is_empty]  # Remove empty results

def save_to_kml(polygons, output_kml):
    """
    Save a list of Shapely Polygon objects to a KML file.

    :param polygons: List of Shapely Polygon objects.
    :param output_kml: The path to the output KML file.
    """
    # Create the KML structure
    kml = etree.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    document = etree.SubElement(kml, "Document")

    # Iterate over each polygon and add it to the KML document
    for idx, polygon in enumerate(polygons):
        # Prepare coordinates in KML format (longitude, latitude, and altitude as 0)
        kml_coordinates = " ".join([f"{lon},{lat},0" for lon, lat in polygon.exterior.coords])

        # Create a Placemark for the Polygon
        placemark = etree.SubElement(document, "Placemark")
        placemark_name = etree.SubElement(placemark, "name")
        placemark_name.text = f"Polygon {idx + 1}"

        # Add the Polygon geometry to the Placemark
        polygon_element = etree.SubElement(placemark, "Polygon")
        outer_boundary_is = etree.SubElement(polygon_element, "outerBoundaryIs")
        linear_ring = etree.SubElement(outer_boundary_is, "LinearRing")

        # Add the coordinates to the LinearRing
        coordinates = etree.SubElement(linear_ring, "coordinates")
        coordinates.text = kml_coordinates

     # Save the KML to the output file
    with open(output_kml, "wb") as f:
        f.write(etree.tostring(kml, pretty_print=True))

    print(f"KML file saved as '{output_kml}'")



def main():
    """Main function to parse arguments and process the files."""
    parser = argparse.ArgumentParser(description="Clip KML polygons using a JSON WKT bounds polygon.")
    parser.add_argument("--footprint_kml", help="Path to the input KML file.", default='footprint.kml')
    parser.add_argument("--site_json", help="Path to the JSON file containing the clipping bounds in WKT format.", default='site.json')
    parser.add_argument("--ground_ply", help="Path to the ground point cloud file.",default='ground.ply')
    parser.add_argument("--out_dir", help="Path to the ground point cloud file.",default='output')

    
    args = parser.parse_args()
    site_polygon = Path(args.out_dir, "site_polygon.kml").as_posix()
    # Read input files
    kml_polygons = read_kml(args.footprint_kml)
    clip_polygon = read_json(args.site_json)
    # Clip polygons
    clipped_polygons = clip_kml_polygons(kml_polygons, clip_polygon)

    # Save results
    if clipped_polygons:
        save_to_kml(clipped_polygons, site_polygon)
        print(f"Clipped KML saved as {site_polygon}")
    else:
        print("No overlapping polygons found.")
        return

    output_dir = Path(args.out_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    config=lm.REG_CONFIG()
    config.sem_label_type='coco'
    config.src_path=args.ground_ply
    config.out_dir=args.out_dir
    config.footprint_path=site_polygon
    labels, scores,transformations = lm.line_based_matching_g2f(config)


    result=cluster_transforms_dbscan(labels, scores, transformations)

    #print(result)
    # Get the first vertex (lat, lon)
    first_vertex = clipped_polygons[0].exterior.coords[0]  # (lon, lat)

    # Convert to (lat, lon) format
    latitude, longitude = first_vertex[1], first_vertex[0]
    epsg=get_utm_epsg(latitude, longitude)
    data={
        "utmepsg": epsg,
        "transformations": result
    }
    #print(json.dumps(data, indent=4))
    output_json = Path(args.out_dir, "output.json").as_posix()
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
