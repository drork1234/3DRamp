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
