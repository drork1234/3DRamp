from imathnumpy import arrayToNumpy
import numpy as np
import cask
import os


def extract_mesh_frame_data(polymesh_sample):
    """
        This function tries to extract the vertex and face indices information from a PolyMesh frame (sample)
        This function assumes that
    :param polymesh_sample: an IPolyMeshSchameSample that contains the vertex and face indices of the frame
    :raises ValueError: when the mesh faces do not contain the same number of indices
    :return: Tuple[ np.ndarray(#num_vertices, #dim[xyz]),
                    np.ndarray(#num_faces, #num_verts_in_face)]:
            a tuple containing the frame mesh vertices and face indices
    """
    # extract all the vertex positions from the frame (#num_vertices, #dim[xyz])
    mesh_verts = arrayToNumpy(polymesh_sample.getPositions())

    # extract the mesh face counts from the frame
    # NOTE: PyAlembic represents the face indices as a flat array (1 dimension array)
    # and in order to let the programmer know which indices belong to each face, the "faceCounts" array is utilized
    # the faceCounts array is a flat array that tells the programmer how much indices each face contains
    # so suggest we have the following:
    # face_indices = [0 3 4 2 1 4 5 3 2]
    # face_counts = [3 3 3]
    # so using the face_counts array we know we have 3 faces (len(face_counts)), each of them containing 3 vertices
    # and the faces will be:
    # face_0 = [0 3 4]
    # face_1 = [2 1 4]
    # face_2 = [5 3 2]
    mesh_face_counts = arrayToNumpy(polymesh_sample.getFaceCounts())

    # check that all the faces have the same number of vertices
    # since we try to reshape the face_indices array to be in a shape of (#num_faces, #num_vertices_in_face),
    # each face must contain the same number of vertices
    if np.unique(mesh_face_counts).size > 1:
        raise ValueError("Mesh faces are not homogeneous, i.e not all the mesh faces contain the same No. of vertices")

    # create the face indices tensor (#num_faces, #num_vertices_in_face)
    mesh_face_indices = arrayToNumpy(polymesh_sample.getFaceIndices()).reshape((-1, mesh_face_counts[0]))

    # return the frame vertices and face indices
    return mesh_verts, mesh_face_indices


def extract_animation_data(polymesh):
    """
        This function extracts the mesh animation vertices over frames (as a single packed np.ndarray) and the mesh
        face indices.

        This function assumes that the shape is rigid, a.k.a the vertices positions will change over time, but the face
        (and hence the face indices) will not, so it extracts only the first frame face indices while loading all the
        frames' vertices

    :param polymesh: PolyMesh instance that contains the samples (frames) of the mesh animation
    :return: Tuple[ np.ndarray(#num_frames, #num_vertices, #dim[xyz]),
                    np.ndarray(#num_faces, #num_verts_in_face)]:
                a tuple containing the animation mesh vertices and face indices
    """
    # get the samples (frames) of the mesh animation
    mesh_animation_frames = polymesh.samples
    if len(mesh_animation_frames) < 1:
        raise ValueError("Mesh must contain at least one frame in order to extract data!")

    # this function assumes that the body is rigid, a.k.a the vertices positions will change over time, but the
    # face (and the face indices) will not, and will be constructed by the same vertices, so we extract only the first
    # frame face indices
    _, mesh_face_indices = extract_mesh_frame_data(mesh_animation_frames[0])

    # extract each frame vertices and stack them into a 3D tensor (#num_frames, #num_vertices, #dim[xyz])
    mesh_animation_frames_verts = np.array([extract_mesh_frame_data(sample)[0] for sample in mesh_animation_frames])

    return mesh_animation_frames_verts, mesh_face_indices


def save_alembic_animation_to_npy(polymesh, verts_path=None, face_indices_path=None):
    """
        This function takes a PolyMesh object and save its animation frame vertices and mesh face indices to .npy files

    :param polymesh: PolyMesh instance. Its animation vertices and face indices will be saved to .npy files
    :param verts_path: String. The path for the animation frames vertices to be saved
    :param face_indices_path: String. The path for the mesh face indices to be saved
    :return: None
    """
    # extract the mesh animation vertices and mesh indices
    mesh_animation_frame_verts, mesh_indices = extract_animation_data(polymesh)

    # if the path is not given, the vertices will be saved to the current directory
    # set the vertices file path
    if not verts_path:
        verts_path = os.path.join(os.getcwd(), "verts.npy")

    # set the face indices file path
    if not face_indices_path:
        face_indices_path = os.path.join(os.getcwd(), "face_indices.npy")

    # save the tensors to .npy files using numpy
    np.save(verts_path, mesh_animation_frame_verts)
    np.save(face_indices_path, mesh_indices)


if __name__ == "__main__":
    # load an alembic archive from .abc file
    arch = cask.Archive("data/alembic/Running_T-pose.abc")

    # extract a PolyMesh called "BodyShape" from the archive
    shapes = cask.find(arch.top, "BodyShape")
    bodyshape = shapes[0]

    # save its animation vertices and face indices to numpy array files (.npy)
    save_alembic_animation_to_npy(bodyshape)

