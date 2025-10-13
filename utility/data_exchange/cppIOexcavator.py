import numpy as np
import numpy.typing as npt
import struct
import os
import re  # reserved for future use
from typing import List, Dict, Optional, Union, Tuple, TypedDict, cast, Any, Iterable

# --- Type Definitions ---

FloatVal = np.float64  # Default float type for NumPy arrays created from text data

class MatrixContainer(TypedDict):
    data: npt.NDArray[FloatVal]  # Final state for returned data
    rows: int
    cols: int

class _MatrixContainerBuilding(TypedDict):  # Intermediate state during parsing
    data: List[List[float]]  # Data is collected as list of lists first
    rows: int
    cols: int

MappingTable = Dict[str, Union[int, float, str]]  # Keys from file will be strings "0", "1" etc. or actual names

class DatFileData(TypedDict):
    SCALARS: Dict[str, float]  # Uppercase to match section
    ORIGIN_CONTAINER: MatrixContainer  # Uppercase to match section
    FACE_TO_GRID_INDEX_CONTAINER: MatrixContainer  # Uppercase to match section
    VERT_TO_GRID_INDEX_CONTAINER: MatrixContainer  # NEW
    FACE_TYPE_MAP: MappingTable  # Uppercase to match section
    TYPE_COUNT_MAP: MappingTable  # Uppercase to match section
    VERT_TYPE_MAP: MappingTable  # Uppercase to match section
    dat_segment_count: Optional[int]  # These are not sections, keep as is
    dat_segment_binary_filename: Optional[str]  # These are not sections, keep as is

SegmentElementType = Union[np.float32, np.float64, np.int32]

# central type-id → dtype map (little endian)
_TYPE_ID_TO_DTYPE = {
    0: np.dtype('<f4'),  # float32
    1: np.dtype('<f8'),  # float64
    2: np.dtype('<i4'),  # int32
}

# Map numpy dtype -> element_type_id in your C++ format
_DTYPE_TO_TYPE_ID = {
    np.dtype(np.float32): 0,
    np.dtype(np.float64): 1,
    np.dtype(np.int32):   2,
}

Segment = npt.NDArray[SegmentElementType]

class FullSegmentationData(TypedDict):
    SCALARS: Dict[str, float]
    ORIGIN_CONTAINER: MatrixContainer
    FACE_TO_GRID_INDEX_CONTAINER: MatrixContainer
    VERT_TO_GRID_INDEX_CONTAINER: MatrixContainer  # NEW
    FACE_TYPE_MAP: MappingTable
    TYPE_COUNT_MAP: MappingTable
    VERT_TYPE_MAP: MappingTable
    dat_segment_count: Optional[int]
    dat_segment_binary_filename: Optional[str]
    segment_container: List[Segment]

# Centralized list of matrix sections we parse the same way
MATRIX_SECTIONS = (
    'ORIGIN_CONTAINER',
    'FACE_TO_GRID_INDEX_CONTAINER',
    'VERT_TO_GRID_INDEX_CONTAINER',
)

# --- Parsing Functions ---

def parse_dat_file(dat_filepath: str) -> Optional[DatFileData]:
    """
    Parses the V4.0 format .dat text file.
    Returns a dictionary adhering to DatFileData TypedDict,
    including metadata for the binary segment file.
    """
    # Initialize building structure
    _dat_data_building: Dict[str, Any] = {
        'SCALARS': {},
        'FACE_TYPE_MAP': {},
        'TYPE_COUNT_MAP': {},
        'VERT_TYPE_MAP': {},
        # Matrix sections initialized programmatically
        **{
            key: cast(_MatrixContainerBuilding, {'data': [], 'rows': 0, 'cols': 0})
            for key in MATRIX_SECTIONS
        },
    }

    current_section: Optional[str] = None
    parsed_segment_count: Optional[int] = None
    parsed_segment_binary_filename: Optional[str] = None

    line_idx: int = -1
    line_content_for_error: str = ""

    try:
        with open(dat_filepath, 'r') as f:
            lines: List[str] = f.readlines()
    except FileNotFoundError:
        print(f"Error: .dat file not found at {dat_filepath}")
        return None
    except Exception as e:
        print(f"Error opening .dat file {dat_filepath}: {e}")
        return None

    try:
        for line_idx, line_content_for_error in enumerate(lines):
            line: str = line_content_for_error.strip()

            if not line or line.startswith('# Data Container Dump'):
                continue

            # Non-section metadata
            if line.startswith('segment_container_count:'):
                try:
                    parsed_segment_count = int(line.split(':', 1)[1].strip())
                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not parse 'segment_container_count' at line {line_idx + 1}: {e}")
                continue
            if line.startswith('segment_binary_file:'):
                try:
                    parsed_segment_binary_filename = line.split(':', 1)[1].strip()
                except IndexError as e:
                    print(f"Warning: Could not parse 'segment_binary_file' at line {line_idx + 1}: {e}")
                continue
            if line.startswith('#'):
                continue

            # Section headers: set/reset current_section
            if line.startswith('[SCALARS]'):
                current_section = 'SCALARS'
                continue
            if line.startswith('[END_SCALARS]'):
                current_section = None
                continue

            # Matrix sections begin
            if line.startswith('[ORIGIN_CONTAINER]'):
                current_section = 'ORIGIN_CONTAINER'
                continue
            if line.startswith('[END_ORIGIN_CONTAINER]'):
                current_section = None
                continue

            if line.startswith('[FACE_TO_GRID_INDEX_CONTAINER]'):
                current_section = 'FACE_TO_GRID_INDEX_CONTAINER'
                continue
            if line.startswith('[END_FACE_TO_GRID_INDEX_CONTAINER]'):
                current_section = None
                continue

            if line.startswith('[VERT_TO_GRID_INDEX_CONTAINER]'):
                current_section = 'VERT_TO_GRID_INDEX_CONTAINER'
                continue
            if line.startswith('[END_VERT_TO_GRID_INDEX_CONTAINER]'):
                current_section = None
                continue
            # Matrix sections end

            if line.startswith('[FACE_TYPE_MAP]'):
                current_section = 'FACE_TYPE_MAP'
                continue
            if line.startswith('[END_FACE_TYPE_MAP]'):
                current_section = None
                continue

            if line.startswith('[TYPE_COUNT_MAP]'):
                current_section = 'TYPE_COUNT_MAP'
                continue
            if line.startswith('[END_TYPE_COUNT_MAP]'):
                current_section = None
                continue

            if line.startswith('[VERT_TYPE_MAP]'):
                current_section = 'VERT_TYPE_MAP'
                continue
            if line.startswith('[END_VERT_TYPE_MAP]'):
                current_section = None
                continue

            # --- Section-specific parsing ---
            if current_section == 'SCALARS':
                try:
                    key, value = line.split(':', 1)
                    cast(Dict[str, float], _dat_data_building['SCALARS'])[key.strip()] = float(value.strip())
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse scalar: '{line}'. Error: {e}")

            elif current_section in MATRIX_SECTIONS:
                container_ref = cast(_MatrixContainerBuilding, _dat_data_building[current_section])
                if line.startswith('rows:'):
                    try:
                        container_ref['rows'] = int(line.split(':', 1)[1].strip())
                    except (ValueError, IndexError):
                        print(f"Warning: Invalid 'rows': {line}")
                elif line.startswith('cols:'):
                    try:
                        container_ref['cols'] = int(line.split(':', 1)[1].strip())
                    except (ValueError, IndexError):
                        print(f"Warning: Invalid 'cols': {line}")
                elif container_ref.get('rows', 0) > 0:
                    # Accept row lines (space separated floats)
                    try:
                        row_vals = list(map(float, line.split()))
                        if len(container_ref['data']) < container_ref['rows']:
                            container_ref['data'].append(row_vals)
                    except ValueError:
                        print(f"Warning: Non-float data in matrix: '{line}'")

            elif current_section in ['FACE_TYPE_MAP', 'TYPE_COUNT_MAP', 'VERT_TYPE_MAP']:
                map_ref = cast(MappingTable, _dat_data_building[current_section])
                if not line.startswith('count:'):
                    try:
                        key_from_line, value_from_line = line.split(':', 1)
                        key_stripped = key_from_line.strip()
                        value_stripped = value_from_line.strip()
                        processed_value: Union[int, float, str]
                        try:
                            processed_value = int(value_stripped)
                        except ValueError:
                            try:
                                processed_value = float(value_stripped)
                            except ValueError:
                                processed_value = value_stripped
                        map_ref[key_stripped] = processed_value
                    except ValueError:
                        print(f"Warning: Malformed map line: '{line.strip()}'")

    except Exception as e:
        print(
            f"Critical error during .dat file parsing at line {line_idx + 1} ('{line_content_for_error.strip()}'): {e}"
        )
        import traceback; traceback.print_exc()
        return None

    # --- Convert collected list data in matrices to NumPy arrays ---
    final_dat_data = cast(DatFileData, _dat_data_building)

    for container_key_str in MATRIX_SECTIONS:
        building_container = cast(_MatrixContainerBuilding, _dat_data_building[container_key_str])
        final_container_typed = cast(MatrixContainer, final_dat_data[container_key_str])

        list_of_rows: List[List[float]] = building_container['data']
        parsed_rows: int = building_container['rows']
        parsed_cols: int = building_container['cols']

        if list_of_rows:
            try:
                np_matrix: npt.NDArray[FloatVal] = np.array(list_of_rows, dtype=FloatVal)
                if np_matrix.ndim == 2 and np_matrix.shape == (parsed_rows, parsed_cols):
                    final_container_typed['data'] = np_matrix
                elif np_matrix.ndim == 1 and parsed_rows == 1 and np_matrix.shape[0] == parsed_cols:
                    final_container_typed['data'] = np_matrix.reshape(1, parsed_cols)
                elif parsed_rows == 0 and parsed_cols == 0 and np_matrix.size == 0:
                    final_container_typed['data'] = np.empty((0, 0), dtype=FloatVal)
                else:
                    print(
                        f"Warning: Shape issue for {container_key_str}. Parsed {parsed_rows}x{parsed_cols}, "
                        f"got array shape {np_matrix.shape}. Using inferred array."
                    )
                    final_container_typed['data'] = np_matrix
            except Exception as e:
                print(
                    f"Warning: Could not convert {container_key_str} data to NumPy array: {e}. "
                    f"Storing empty array with parsed dims."
                )
                final_container_typed['data'] = np.empty((parsed_rows, parsed_cols), dtype=FloatVal)
        else:
            final_container_typed['data'] = np.empty((parsed_rows, parsed_cols), dtype=FloatVal)

        final_container_typed['rows'] = parsed_rows
        final_container_typed['cols'] = parsed_cols

    final_dat_data['dat_segment_count'] = parsed_segment_count
    final_dat_data['dat_segment_binary_filename'] = parsed_segment_binary_filename

    return final_dat_data

# ---------- Generic core reader + typed wrappers ----------

def _load_arrays_from_binary(
    bin_filepath: str,
    expected_segment_count_from_dat: Optional[int] = None,
    allowed_type_ids: Optional[Iterable[int]] = None,
) -> Optional[List[Segment]]:
    """
    Generic reader for your .bin format:
      <uint32 magic=0x5345474D, uint16 version=1, uint8 type_id, uint32 num_segments>
      num_segments * <uint32 d0,d1,d2>
      then payloads (row-major) per segment
    If allowed_type_ids is given, enforce element_type_id ∈ allowed_type_ids.
    """
    segments_list: List[Segment] = []
    if not os.path.exists(bin_filepath):
        print(f"Error: Binary file not found: {bin_filepath}")
        return None

    try:
        with open(bin_filepath, 'rb') as bf:
            header_format = '<IHBI'
            header_size = struct.calcsize(header_format)
            header = bf.read(header_size)
            if len(header) < header_size:
                raise IOError(f"{bin_filepath} too short for header")
            magic, version, type_id, num_segments = struct.unpack(header_format, header)

            if magic != 0x5345474D:
                raise ValueError(f"Bad magic: {magic:#x}")
            if version != 1:
                raise ValueError(f"Unsupported version: {version}")

            if expected_segment_count_from_dat is not None and expected_segment_count_from_dat != num_segments:
                print(f"Warning: .dat count {expected_segment_count_from_dat} != .bin count {num_segments}. Using .bin.")

            if allowed_type_ids is not None and type_id not in set(allowed_type_ids):
                raise ValueError(f"Unexpected element_type_id {type_id}; allowed {sorted(set(allowed_type_ids))}")

            if type_id not in _TYPE_ID_TO_DTYPE:
                raise ValueError(f"Unknown element_type_id: {type_id}")

            np_dtype = _TYPE_ID_TO_DTYPE[type_id]

            # read descriptors
            desc_fmt = '<III'
            desc_sz = struct.calcsize(desc_fmt)
            dims_list: List[Tuple[int, int, int]] = []
            for i in range(num_segments):
                desc = bf.read(desc_sz)
                if len(desc) < desc_sz:
                    raise IOError(f"EOF reading descriptor {i}/{num_segments}")
                dims_list.append(cast(Tuple[int,int,int], struct.unpack(desc_fmt, desc)))

            # read payloads
            for i, dims in enumerate(dims_list):
                d0, d1, d2 = dims
                num_elems = int(np.prod(dims))
                if num_elems == 0:
                    segments_list.append(np.empty((d0, d1, d2), dtype=np_dtype))
                    continue

                nbytes = num_elems * np_dtype.itemsize
                buf = bf.read(nbytes)
                if len(buf) < nbytes:
                    raise IOError(f"Segment {i}: expected {nbytes} bytes, got {len(buf)}")

                flat = np.frombuffer(buf, dtype=np_dtype, count=num_elems)
                if flat.size != num_elems:
                    raise IOError(f"Segment {i}: expected {num_elems} elems, got {flat.size}")
                segments_list.append(flat.reshape((d0, d1, d2)))

    except Exception as e:
        print(f"Error processing {bin_filepath}: {e}")
        import traceback; traceback.print_exc()
        return None

    return segments_list

def load_segments_from_binary(
    bin_filepath: str,
    expected_segment_count_from_dat: Optional[int] = None
) -> Optional[List[Segment]]:
    # segments may be float32 (0) or float64 (1)
    return _load_arrays_from_binary(
        bin_filepath,
        expected_segment_count_from_dat=expected_segment_count_from_dat,
        allowed_type_ids=(0, 1),
    )

def load_labels_from_binary(
    bin_filepath: str,
    expected_segment_count_from_dat: Optional[int] = None
) -> Optional[List[Segment]]:
    # labels are int32 (2)
    return _load_arrays_from_binary(
        bin_filepath,
        expected_segment_count_from_dat=expected_segment_count_from_dat,
        allowed_type_ids=(2,),
    )

def load_predictions_from_binary(
    bin_filepath: str,
    expected_segment_count_from_dat: Optional[int] = None
) -> Optional[List[Segment]]:
    # predictions are int32 (2)
    return _load_arrays_from_binary(
        bin_filepath,
        expected_segment_count_from_dat=expected_segment_count_from_dat,
        allowed_type_ids=(2,),
    )

def _resolve_dtype(dtype: Union[np.dtype, type, str]) -> np.dtype:
    dt = np.dtype(dtype)
    # Force little-endian to match the on-disk format
    if dt.byteorder not in ('<', '=', '|'):  # '=' is native-endian: on little-endian machines it's fine
        dt = dt.newbyteorder('<')
    if dt.kind in ('f', 'i'):
        # normalize sizes (e.g., float64 vs '<f8')
        dt = np.dtype(dt.str.replace('=', '<'))  # ensure '<'
    return dt

# ---------- Generic core writer + typed wrappers ----------

def _common_float_dtype(arrays: Iterable[np.ndarray]) -> np.dtype:
    """
    Infer a common *float* dtype across arrays.
    Returns <f8 if any float64 present, else <f4 if any float32 present.
    Errors on non-float or empty input.
    """
    has_f8 = False
    has_f4 = False
    seen_any = False
    for a in arrays:
        if not isinstance(a, np.ndarray):
            raise TypeError("Inputs must be numpy arrays.")
        if a.ndim != 3:
            raise ValueError(f"All arrays must be 3D; got {a.shape}.")
        seen_any = True
        k = np.dtype(a.dtype).kind
        sz = np.dtype(a.dtype).itemsize
        if k != 'f':
            raise ValueError(f"Found non-float dtype {a.dtype} in segments; "
                             "segments must be float arrays.")
        if sz == 8:
            has_f8 = True
        elif sz == 4:
            has_f4 = True
        else:
            raise ValueError(f"Unsupported float dtype {a.dtype}; use float32/float64.")
    if not seen_any:
        # default to float32 for an empty set, still write a valid file
        return np.dtype('<f4')
    return np.dtype('<f8') if has_f8 else np.dtype('<f4')


def _save_arrays_to_binary(
    bin_filepath: str,
    arrays: Iterable[np.ndarray],
    dtype: Union[np.dtype, type, str],
    format_version: int = 1,
) -> None:
    """
    Core binary writer compatible with your C++ loader.

    - All arrays must be 3D.
    - `dtype` controls on-disk element_type_id (float32/float64/int32 only).
    - Arrays are cast to `dtype`, made C-contiguous & little-endian before writing.
    """
    arr_list: List[np.ndarray] = list(arrays)
    target_dtype = _resolve_dtype(dtype)  # uses your existing helper

    # Normalize to canonical LE dtype keys
    normalized_key = np.dtype(target_dtype).newbyteorder('<')
    if normalized_key not in _DTYPE_TO_TYPE_ID:
        raise ValueError("Unsupported dtype. Allowed: float32, float64, int32.")

    element_type_id = _DTYPE_TO_TYPE_ID[normalized_key]

    # Prepare shapes and contiguous buffers
    shapes: List[Tuple[int, int, int]] = []
    contiguous: List[np.ndarray] = []
    for idx, arr in enumerate(arr_list):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Array {idx} is not a numpy array.")
        if arr.ndim != 3:
            raise ValueError(f"Array {idx} must be 3D; got {arr.shape}.")
        # cast to target dtype and ensure C-contiguous, little-endian
        ac = np.ascontiguousarray(arr.astype(target_dtype, copy=False))
        if ac.dtype.byteorder not in ('<', '|'):
            ac = ac.byteswap().newbyteorder('<')
        d0, d1, d2 = map(int, ac.shape)
        for d, name in zip((d0, d1, d2), ('d0', 'd1', 'd2')):
            if d < 0 or d > 0xFFFFFFFF:
                raise ValueError(f"Array {idx} dimension {name}={d} out of uint32 range.")
        shapes.append((d0, d1, d2))
        contiguous.append(ac)

    # Write file
    with open(bin_filepath, 'wb') as bf:
        magic = 0x5345474D  # 'SEGM'
        num_segments = len(arr_list)
        bf.write(struct.pack('<IHBI', magic, format_version, element_type_id, num_segments))
        for d0, d1, d2 in shapes:
            bf.write(struct.pack('<III', d0, d1, d2))
        for ac in contiguous:
            if 0 in ac.shape:
                continue
            bf.write(ac.tobytes(order='C'))


def save_segments(
    bin_filepath: str,
    segments: Iterable[np.ndarray],
    dtype: Union[np.dtype, type, str, None] = None,
    format_version: int = 1,
) -> None:
    """
    Segments → float arrays. If dtype is None, infer a common float dtype across inputs:
      - float64 if any array is float64, else float32.
    """
    seg_list = list(segments)
    if dtype is None:
        inferred = _common_float_dtype(seg_list)
        _save_arrays_to_binary(bin_filepath, seg_list, inferred, format_version=format_version)
    else:
        # Enforce the provided dtype is float
        dt = np.dtype(dtype)
        if dt.kind != 'f' or dt.itemsize not in (4, 8):
            raise ValueError("save_segments dtype must be float32 or float64.")
        _save_arrays_to_binary(bin_filepath, seg_list, dt, format_version=format_version)


def save_labels(
    bin_filepath: str,
    labels: Iterable[np.ndarray],
    format_version: int = 1,
) -> None:
    """
    Labels → int arrays (int32 on disk).
    Non-int inputs will be safely cast to int32.
    """
    _save_arrays_to_binary(bin_filepath, list(labels), np.int32, format_version=format_version)


def save_predictions(
    bin_filepath: str,
    predictions: Iterable[np.ndarray],
    format_version: int = 1,
) -> None:
    """
    Predictions → int arrays (int32 on disk).
    """
    _save_arrays_to_binary(bin_filepath, list(predictions), np.int32, format_version=format_version)



def main():
    labels = load_labels_from_binary(r"H:\ws_seg_debug\debug_output\00000004\segmentation_data_labels.bin")
    segments = load_segments_from_binary(r"H:\ws_seg_debug\debug_output\00000004\segmentation_data_segments.bin")


    print()



if __name__ == '__main__':

    main()
