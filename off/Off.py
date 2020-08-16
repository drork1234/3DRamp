import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from itertools import product
from off.importer import read_off, write_off
from scipy.sparse import csr_matrix
from types import LambdaType


class OffMesh:
    def __init__(self, off_f_path: str = None):
        self.vertices: np.ndarray = None
        self.faces: np.ndarray = None
        self.adj_matrix: csr_matrix = None
        self.pv_mesh: pv.PolyData = None
        if off_f_path:
            self.load_mesh(off_f_path)

    def load_mesh(self, off_f_path: str) -> None:
        mesh = read_off(off_f_path=off_f_path)
        self.vertices, self.faces = mesh["vertices"], mesh["faces"]
        self.pv_mesh = pv.PolyData(self.vertices, self.faces)
        self.__create_vertex_adj_mat__()

    def save_mesh(self, off_f_path: str) -> None:
        if self.vertices is None or self.faces is None:
            raise ValueError("Vertices or faces do not exist, can't save to OFF file!")

        write_off(off_f_path=off_f_path, vertices=self.vertices, faces=self.faces)

    def __create_vertex_adj_mat__(self):
        if self.faces is None:
            raise ValueError("Vertices or faces do not exist, can't create an adjacency matrix!")

        self.adj_matrix = np.zeros(shape=(self.vertices.shape[0], self.vertices.shape[0]), dtype=int)
        for vert_idxes in self.faces[:, 1:]:
            vert_comb = list(product(vert_idxes.tolist(), vert_idxes.tolist()))
            vert_rows, vert_cols = np.split(np.array(vert_comb).transpose(), 2)
            self.adj_matrix[vert_rows, vert_cols] = 1

        # a vertex in a mesh cannot be connected to itself, so main diagonal contains only zeros
        np.fill_diagonal(self.adj_matrix, 0)
        self.adj_matrix = csr_matrix(self.adj_matrix)

    def plot_wireframe(self):
        if self.pv_mesh is None:
            raise ValueError("PyVista OFF Mesh is None. Can't plot!")

        self.pv_mesh.plot(style='wireframe')

    def plot_vertices(self, f: LambdaType):
        if self.pv_mesh is None:
            raise ValueError("PyVista OFF Mesh is None. Can't plot!")

        scalars = f(self.vertices)
        if scalars.shape[0] != self.vertices.shape[0]:
            raise ValueError("PyVista OFF Mesh: scalar.shape[0] {} != #vertices {}".format(scalars.shape[0],
                                                                                           self.vertices[0]))

        self.pv_mesh.plot(style='points', scalars=scalars)

    def plot_faces(self, f: LambdaType):
        if self.pv_mesh is None:
            raise ValueError("PyVista OFF Mesh is None. Can't plot!")

        scalars = f(self.vertices)
        if scalars.shape[0] != self.vertices.shape[0]:
            raise ValueError("PyVista OFF Mesh: scalar.shape[0] {} != #vertices {}".format(scalars.shape[0],
                                                                                           self.vertices[0]))

        self.pv_mesh.plot(style='surface', scalars=scalars)

    def __compute_valence__(self) -> np.ndarray:
        return sum(self.adj_matrix)[0]

    @property
    def valence(self) -> np.ndarray:
        return self.__compute_valence__()

    @property
    def barycenters(self) -> np.ndarray:
        face_verts = self.get_faces(self.faces[:, 1:])
        return face_verts.mean(axis=1)

    @property
    def normals(self) -> np.ndarray:
        face_verts = self.get_faces(self.faces[:, 1:])

        # split the face vertices between the first vertex and the 2 other vertices of the face
        v0, vs = face_verts[:, 0, :], face_verts[:, 1:, :]
        v0 = np.expand_dims(v0, axis=1)

        # to find the face vectors, subtract the two vertices from the first vertex (utilizing broadcasting of v0)
        face_vectors = vs - v0

        # split the matrix to list of 2 vectors
        face_vectors_lst = np.split(face_vectors, 2, axis=1)

        # return the cross product between the vectors
        return np.cross(*face_vectors_lst).squeeze()

    @property
    def areas(self) -> np.ndarray:
        return 0.5 * np.linalg.norm(self.normals, axis=1)

    def vnormals(self, idx: int) -> np.ndarray:
        if idx < 0 or idx > self.vertices.shape[0]:
            raise ValueError("OFF Mesh: vertex index ({}) out of bounds. #vertices is {}!".format(idx,
                                                                                                  self.vertices.shape[0]
                                                                                                  ))

        # get the faces idxs that each vertex belongs to
        # face rows are the indices, so the column 0
        adj_faces_idxs = np.argwhere(self.faces[:, 1:] == idx)[:, 0]

        # get the area for each adjacent face
        adj_face_ares = self.areas[adj_faces_idxs]

        # get the normal for each adjacent face
        adj_face_norms = self.normals[adj_faces_idxs]

        # normalize the face vectors
        adj_face_norms = adj_face_norms / np.linalg.norm(adj_face_norms, axis=1).reshape((-1, 1))
        print(np.linalg.norm(adj_face_norms, axis=1))

        # create a weighted vector of the face areas and the face normals
        vert_normal = np.sum(adj_face_ares[:, np.newaxis] * adj_face_norms, axis=0)

        # normalize the weighted vector
        vert_normal = vert_normal / np.linalg.norm(vert_normal)

        return vert_normal

    def curvature(self, idx: int) -> float:
        if idx < 0 or idx > self.vertices.shape[0]:
            raise ValueError("OFF Mesh: vertex index ({}) out of bounds. #vertices is {}!".format(idx,
                                                                                                  self.vertices.shape[0]
                                                                                                  ))
        # get all vertices adjacent to the requested vertex according to their faces
        # 1. get the all the faces
        faces_idxes = self.faces[:, 1:]

        # 2. get all faces indices where the vertex is a part of the face
        vert_face_row_occurs = np.argwhere(faces_idxes == idx)[:, 0]

        # 3. get faces in which the vertex is a part of the face
        vert_faces_idxs = faces_idxes[vert_face_row_occurs]

        # 4. get the number of faces in which the vertex is a part of
        n_vert_occurs = vert_face_row_occurs.shape[0]

        # 5. remove the index of the vertex from the face-vertices array, to include the other remaining vertices of the face
        other_face_verts_idxs = vert_faces_idxs[~(vert_faces_idxs == idx)].reshape((n_vert_occurs, -1))
        other_face_verts = self.vertices[other_face_verts_idxs.reshape(-1)].reshape((n_vert_occurs, 2, 3))

        # compute the vectors according to the vertex for each face
        this_vert = np.expand_dims(self.vertices[idx][np.newaxis, :], axis=0)
        faces_vectors = other_face_verts - this_vert

        # compute the cosine similarity between the vectors, which is the angle of the vertex between the 2 vectors
        vert_cosines = np.sum(faces_vectors[:, 0, :] * faces_vectors[:, 1, :], axis=1) / \
                       (np.linalg.norm(faces_vectors[:, 0, :], axis=1) * np.linalg.norm(faces_vectors[:, 1, :], axis=1))

        # compute the angles of the vertex
        vert_angles = np.arccos(vert_cosines)

        # compute the curvature of the vertex: 2*PI - sum_of_angles_near_the_vertex
        vert_curvature = 2*np.pi - np.sum(vert_angles)

        return vert_curvature

    def get_faces(self, face_idxs: np.ndarray) -> np.ndarray:
        verts = self.vertices[face_idxs.reshape(-1)]
        return verts.reshape((-1, 3, 3))
