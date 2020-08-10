import os
from typing import Dict, List, Tuple
from off.importer import read_off, write_off


class OffMesh:
    def __init__(self, off_f_path: str = None):
        self.vertices = []
        self.faces = []
        if off_f_path:
            self.load_mesh(off_f_path)

    def load_mesh(self, off_f_path: str) -> None:
        mesh = read_off(off_f_path=off_f_path)
        self.vertices, self.faces = mesh["vertices"], mesh["faces"]

    def save_mesh(self, off_f_path: str) -> None:
        write_off(off_f_path=off_f_path, vertices=self.vertices, faces=self.faces)


if __name__ == "__main__":
    print("Hello!")