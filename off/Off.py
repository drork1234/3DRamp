import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from itertools import product
from off.importer import read_off, write_off
from scipy.sparse import csr_matrix, lil_matrix
from types import LambdaType
from typing import Dict, List
from tqdm import tqdm


class OffMesh:
    def __init__(self, off_f_path: str = None):
        self.vertices: np.ndarray = None
        self.faces: np.ndarray = None
        self.adj_matrix: csr_matrix = None
        self.pv_mesh: pv.PolyData = None
        self.areas: np.ndarray = None
        self.normals: np.ndarray = None
        self.vnormals: np.ndarray = None
        self.vcurvature: np.ndarray = None
        if off_f_path:
            self.load_mesh(off_f_path)

    def load_mesh(self, off_f_path: str) -> None:
        mesh = read_off(off_f_path=off_f_path)
        self.vertices, self.faces = mesh["vertices"], mesh["faces"]
        self.pv_mesh = pv.PolyData(self.vertices, self.faces)
        self.__calculate_mesh_properties__()

    def save_mesh(self, off_f_path: str) -> None:
        if self.vertices is None or self.faces is None:
            raise ValueError("Vertices or faces do not exist, can't save to OFF file!")

        write_off(off_f_path=off_f_path, vertices=self.vertices, faces=self.faces)

    def __calculate_mesh_properties__(self):
        print("Computing vertex-vertex adjacency matrix...")
        self.__create_vertex_adj_mat__()

        print("Computing face normals...")
        self.normals = self.__compute_normals__()

        print("Computing face areas...")
        self.areas = self.__compute_areas__()

        print("Computing vertex normals...")
        self.vnormals = self.__compute_vnormals__()

        print("Computing vertices gaussian curvature...")
        self.vcurvature = self.__compute_curvatures__()

    def __create_vertex_adj_mat__(self):
        if self.faces is None:
            raise ValueError("Vertices or faces do not exist, can't create an adjacency matrix!")

        self.adj_matrix = lil_matrix((self.vertices.shape[0], self.vertices.shape[0]), dtype=int)
        for vert_idxes in self.faces[:, 1:]:
            vert_comb = list(product(vert_idxes.tolist(), vert_idxes.tolist()))
            vert_rows, vert_cols = np.split(np.array(vert_comb).transpose(), 2)
            self.adj_matrix[vert_rows, vert_cols] = 1

        diag = np.arange(0, self.vertices.shape[0], dtype=int)
        # a vertex in a mesh cannot be connected to itself, so main diagonal contains only zeros
        self.adj_matrix[diag, diag] = 0
        self.adj_matrix = csr_matrix(self.adj_matrix)

    def plot_wireframe(self):
        if self.pv_mesh is None:
            raise ValueError("PyVista OFF Mesh is None. Can't plot!")

        self.pv_mesh.plot(style='wireframe')

    def plot_vertices(self, **kwd):
        if self.pv_mesh is None:
            raise ValueError("PyVista OFF Mesh is None. Can't plot!")

        f = kwd.pop("f", None)
        scalars = kwd.pop("scalars", None)
        if f is not None:
            mscalars = f(self.vertices) if scalars is None else f(scalars)
        elif scalars is not None:
            mscalars = scalars
        else:
            raise KeyError("Scalars must be created by a function (given by 'f') or given by the user (by 'scalars')")

        if mscalars.shape[0] != self.vertices.shape[0]:
            raise ValueError("PyVista OFF Mesh: scalar.shape[0] {} != #vertices {}".format(mscalars.shape[0],
                                                                                           self.vertices[0]))

        self.pv_mesh.plot(style='points', scalars=mscalars, scalar_bar_args={"interactive": True}, show_scalar_bar=True)

    def plot_faces(self, **kwd):
        if self.pv_mesh is None:
            raise ValueError("PyVista OFF Mesh is None. Can't plot!")

        f = kwd.pop("f", None)
        scalars = kwd.pop("scalars", None)
        if f is not None:
            mscalars = f(self.vertices) if scalars is None else f(scalars)
        elif scalars is not None:
            mscalars = scalars
        else:
            raise KeyError("Scalars must be created by a function (given by 'f') or given by the user (by 'scalars')")

        if mscalars.shape[0] != self.vertices.shape[0]:
            raise ValueError("PyVista OFF Mesh: scalar.shape[0] {} != #vertices {}".format(mscalars.shape[0],
                                                                                           self.vertices[0]))

        self.pv_mesh.plot(style='surface', scalars=mscalars, scalar_bar_args={"interactive": True}, show_scalar_bar=True)


    def __compute_valence__(self) -> np.ndarray:
        return np.squeeze(sum(self.adj_matrix).toarray())

    @property
    def valence(self) -> np.ndarray:
        return self.__compute_valence__()

    @property
    def barycenters(self) -> np.ndarray:
        face_verts = self.get_faces(self.faces[:, 1:])
        return face_verts.mean(axis=1)

    def __compute_normals__(self) -> np.ndarray:
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

    def __compute_areas__(self) -> np.ndarray:
        return 0.5 * np.linalg.norm(self.normals, axis=1)

    def __compute_vnormals__(self) -> np.ndarray:
        # get the faces idxs that each the vertex belongs to
        # face rows are the indices, so take column 0 (column 1 is the column in the matrix where the vertex index was found)
        # face_groups contains the subgroups of vertices split by the number of faces that they are included in
        # each value in the dict is a dictionary containing the vertex indices and their face indices that they are contained in
        this_valence = self.valence
        all_mesh_valence_vals = np.unique(this_valence)
        vert_normals = np.zeros(shape=(self.vertices.shape[0], 3), dtype=float)

        for valence in all_mesh_valence_vals:
            vert_idxs = np.argwhere(this_valence == valence)
            adj_faces_idxs = np.argwhere(self.faces[:, 1:] == vert_idxs[:, :, np.newaxis])
            adj_faces_idxs = adj_faces_idxs.reshape((vert_idxs.shape[0], valence, -1))[:, :, 1]

            # get the area for each adjacent face
            adj_face_ares = self.areas[adj_faces_idxs]

            # get the normal for each adjacent face
            adj_face_norms = self.normals[adj_faces_idxs]

            # normalize the face vectors
            adj_face_norms = adj_face_norms / np.linalg.norm(adj_face_norms, axis=2, keepdims=True)

            # create a weighted vector of the face areas and the face normals
            vert_normal = np.sum(np.expand_dims(adj_face_ares, axis=2) * adj_face_norms, axis=1)

            # normalize the weighted vector
            vert_normals[np.squeeze(vert_idxs)] = vert_normal / np.linalg.norm(vert_normal, axis=1, keepdims=True)

        return vert_normals

    def __compute_curvatures__(self) -> np.ndarray:
        this_valence = self.valence
        all_mesh_valence_vals = np.unique(this_valence)
        vert_curvs = np.zeros(shape=(self.vertices.shape[0], ), dtype=float)

        for valence in all_mesh_valence_vals:
            vert_idxs = np.argwhere(this_valence == valence)
            # this returns the faces indices in which the vertex is a part of
            adj_faces_idxs = np.argwhere(self.faces[:, 1:] == vert_idxs[:, :, np.newaxis])
            adj_faces_idxs = adj_faces_idxs.reshape((vert_idxs.shape[0], valence, -1))[:, :, 1]

            # get all vertices adjacent to the requested vertex according to their faces
            # 1. get the all the faces
            faces_idxes = self.faces[:, 1:]

            # 2. get faces in which the vertex is a part of the face
            # this returns the faces of which the vertex is a part of
            vert_all_faces_idxs = faces_idxes[adj_faces_idxs]

            # 3. remove the index of the vertex from the face-vertices array, to include the other remaining vertices of the face
            other_faces_vert_idxs = vert_all_faces_idxs[~(vert_all_faces_idxs == vert_idxs[:, :, np.newaxis])].\
                reshape((vert_idxs.shape[0], valence, -1))
            other_faces_vert = self.vertices[other_faces_vert_idxs.reshape(-1)].\
                reshape((vert_idxs.shape[0], valence, 2, 3))

            # 4. compute the vectors according to the vertex for each face
            these_verts = np.expand_dims(self.vertices[vert_idxs], axis=1)
            all_faces_vectors = other_faces_vert - these_verts

            # 5. compute the cosine similarity between the vectors, which is the angle of the vertex between the 2 vectors
            verts_cosines = np.sum(all_faces_vectors[:, :, 0, :] * all_faces_vectors[:, :, 1, :], axis=2) / \
                           (np.linalg.norm(all_faces_vectors[:, :, 0, :], axis=2) * np.linalg.norm(all_faces_vectors[:, :, 1, :], axis=2))

            # 6. compute the angles of the vertex
            # vert_angles = np.arccos(vert_cosines)
            verts_angles = np.arccos(verts_cosines)

            # 7. compute the curvature of the vertex: 2*PI - sum_of_angles_near_the_vertex
            # vert_curvature = 2*np.pi - np.sum(vert_angles)
            verts_curvature = 2*np.pi - np.sum(verts_angles, axis=1)

            # 8. assign the vertices curvature values to the computed vertex indices
            vert_curvs[vert_idxs.squeeze()] = verts_curvature

        return vert_curvs

    def get_faces(self, face_idxs: np.ndarray) -> np.ndarray:
        verts = self.vertices[face_idxs.reshape(-1)]
        return verts.reshape((-1, 3, 3))
