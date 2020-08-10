import os
import numpy as np
import re
from typing import Dict, List, Tuple


def read_off(off_f_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(off_f_path):
        raise FileNotFoundError("{} not found!".format(off_f_path))

    with open(off_f_path, 'r') as f:
        lines = [line.strip().split() for line in f.readlines()]

        # if the 1st line doesn't contain only the word OFF, this is not an OFF file
        if lines[0][0] != 'OFF':
            raise ValueError("{} is not an OFF file")

        # if the 2nd line doesn't contain only 3 numbers, this is not an OFF file
        mesh_meta = list(map(int, lines[1]))
        if len(mesh_meta) != 3:
            raise ValueError("OFF Mesh metadata must conform to the format: (numVertices, numFaces, numEdges)")

        # get the number of verticecs, faces and edges
        num_verts, num_faces, num_edges = mesh_meta

        # check that the number of declared vertices and faces match the number of lines in the file
        # (minus 2 header lines)
        if num_verts + num_faces != len(lines) - 2:  # according to OFF spec, num_edges can be safely ignores
            raise ValueError("OFF mesh file declares {} vertices + edges, but only {} were found".format(num_verts + num_faces, len(lines) - 2))

        # the following 'num_verts' lines in the file contain 3 float vertices per line
        mesh_vertices = np.array(lines[2:2+num_verts]).astype(float)

        # mesh faces are of the format: #N vert_idx0 vert_idx1 ... vert_idxN
        mesh_faces = np.array(lines[2+num_verts:]).astype(int)
        mesh_faces = mesh_faces[:, 1:]

        return {"vertices": mesh_vertices, "faces": mesh_faces}


def write_off(off_f_path: str, vertices: np.ndarray, faces: np.ndarray):
    # top two lines of the OFF file are the header
    # 1st line: 'OFF'
    # 2nd line: num_verts, num_faces, num_edges (0 by default, can be ignored)
    off_header = ["OFF",
                  "{} {} {}".format(vertices.shape[0], faces.shape[0], 0)]

    # the header is followed by the vertex lines
    off_vert_lines_str = np.array2string(vertices)
    # replace [] characters printed as list brackets by array2string
    off_vert_lines_str = re.sub(r'[\[\]]', '', off_vert_lines_str)

    # vertex lines are followed by face vertex index lines
    off_face_lines_str = np.array2string(faces)
    off_face_lines_str = re.sub(r'[\[\]]', '', off_face_lines_str)

    # combine the header, vertices string and faces string
    off_lines = off_header + [off_vert_lines_str, off_face_lines_str]

    # write the lines to the file
    with open(off_f_path, 'w') as f:
        f.write("\n".join(off_lines))


if __name__ == "__main__":
    mesh = read_off("../data/example_off_files/sphere_s0.off")
    write_off("../data/example_off_files/saved_sphere_s0.off", mesh["vertices"], mesh["faces"])
