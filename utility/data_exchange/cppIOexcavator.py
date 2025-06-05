import numpy as np
import numpy.typing as npt
import struct
import os
import re  # For potential future use, though not heavily used now
from typing import List, Dict, Optional, Union, Tuple, TypedDict, cast, Any

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
    FACE_TYPE_MAP: MappingTable  # Uppercase to match section
    TYPE_COUNT_MAP: MappingTable  # Uppercase to match section
    VERT_TYPE_MAP: MappingTable  # Uppercase to match section
    dat_segment_count: Optional[int]  # These are not sections, keep as is
    dat_segment_binary_filename: Optional[str]  # These are not sections, keep as is


SegmentElementType = Union[np.float32, np.float64]  # Type of elements in binary segments
Segment = npt.NDArray[SegmentElementType]


class FullSegmentationData(TypedDict):
    SCALARS: Dict[str, float]
    ORIGIN_CONTAINER: MatrixContainer
    FACE_TO_GRID_INDEX_CONTAINER: MatrixContainer
    FACE_TYPE_MAP: MappingTable
    TYPE_COUNT_MAP: MappingTable
    VERT_TYPE_MAP: MappingTable
    dat_segment_count: Optional[int]
    dat_segment_binary_filename: Optional[str]
    segment_container: List[Segment]


# --- Parsing Functions ---

def parse_dat_file(dat_filepath: str) -> Optional[DatFileData]:
    """
    Parses the V4.0 format .dat text file.
    Returns a dictionary adhering to DatFileData TypedDict,
    including metadata for the binary segment file.
    """
    _dat_data_building: Dict[str, Any] = {
        'SCALARS': {},
        'ORIGIN_CONTAINER': cast(_MatrixContainerBuilding, {'data': [], 'rows': 0, 'cols': 0}),
        'FACE_TO_GRID_INDEX_CONTAINER': cast(_MatrixContainerBuilding, {'data': [], 'rows': 0, 'cols': 0}),
        'FACE_TYPE_MAP': {},
        'TYPE_COUNT_MAP': {},
        'VERT_TYPE_MAP': {}
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

            # Section headers - these set current_section to UPPERCASE strings
            if line.startswith('[SCALARS]'):
                current_section = 'SCALARS'
            elif line.startswith('[END_SCALARS]'):
                current_section = None
            elif line.startswith('[ORIGIN_CONTAINER]'):
                current_section = 'ORIGIN_CONTAINER'
            elif line.startswith('[END_ORIGIN_CONTAINER]'):
                current_section = None
            elif line.startswith('[FACE_TO_GRID_INDEX_CONTAINER]'):
                current_section = 'FACE_TO_GRID_INDEX_CONTAINER'
            elif line.startswith('[END_FACE_TO_GRID_INDEX_CONTAINER]'):
                current_section = None
            elif line.startswith('[FACE_TYPE_MAP]'):
                current_section = 'FACE_TYPE_MAP'
            elif line.startswith('[END_FACE_TYPE_MAP]'):
                current_section = None
            elif line.startswith('[TYPE_COUNT_MAP]'):
                current_section = 'TYPE_COUNT_MAP'
            elif line.startswith('[END_TYPE_COUNT_MAP]'):
                current_section = None
            elif line.startswith('[VERT_TYPE_MAP]'):
                current_section = 'VERT_TYPE_MAP'
            elif line.startswith('[END_VERT_TYPE_MAP]'):
                current_section = None

            # Section-specific parsing
            elif current_section == 'SCALARS':  # current_section is 'SCALARS'
                try:
                    key, value = line.split(':', 1)
                    # Accessing _dat_data_building['SCALARS']
                    cast(Dict[str, float], _dat_data_building['SCALARS'])[key.strip()] = float(value.strip())
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse scalar: '{line}'. Error: {e}")

            elif current_section in ['ORIGIN_CONTAINER', 'FACE_TO_GRID_INDEX_CONTAINER']:
                # current_section is 'ORIGIN_CONTAINER' or 'FACE_TO_GRID_INDEX_CONTAINER' (UPPERCASE)
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
                    if len(container_ref['data']) < container_ref.get('rows', 0):
                        try:
                            container_ref['data'].append(list(map(float, line.split())))
                        except ValueError:
                            print(f"Warning: Non-float data in matrix: '{line}'")

            elif current_section in ['FACE_TYPE_MAP', 'TYPE_COUNT_MAP', 'VERT_TYPE_MAP']:
                # current_section is 'FACE_TYPE_MAP', etc. (UPPERCASE)
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
                    except ValueError:  # Catches split error if no ':'
                        print(f"Warning: Malformed map line: '{line.strip()}'")

    except Exception as e:
        print(
            f"Critical error during .dat file parsing at line {line_idx + 1} ('{line_content_for_error.strip()}'): {e}")
        import traceback;
        traceback.print_exc()
        return None

    # --- Convert collected list data in matrices to NumPy arrays ---
    # _dat_data_building now uses uppercase keys for sections.
    # DatFileData TypedDict also expects uppercase keys for these sections.
    final_dat_data = cast(DatFileData, _dat_data_building)

    for container_key_str in ['ORIGIN_CONTAINER', 'FACE_TO_GRID_INDEX_CONTAINER']:  # Use UPPERCASE keys
        building_container = cast(_MatrixContainerBuilding, _dat_data_building[container_key_str])
        # final_dat_data[container_key_str] should now correctly point to a MatrixContainer-like structure
        # The cast below is to inform the type checker about the structure of this specific part.
        final_container_typed = cast(MatrixContainer, final_dat_data[container_key_str])

        list_of_rows: List[List[float]] = building_container['data']
        parsed_rows: int = building_container['rows']
        parsed_cols: int = building_container['cols']

        if list_of_rows:
            try:
                np_matrix: npt.NDArray[FloatVal] = np.array(list_of_rows, dtype=FloatVal)
                if np_matrix.ndim == 2 and np_matrix.shape[0] == parsed_rows and np_matrix.shape[1] == parsed_cols:
                    final_container_typed['data'] = np_matrix
                elif np_matrix.ndim == 1 and parsed_rows == 1 and np_matrix.shape[0] == parsed_cols:
                    final_container_typed['data'] = np_matrix.reshape(1, parsed_cols)
                elif parsed_rows == 0 and parsed_cols == 0 and np_matrix.size == 0:
                    final_container_typed['data'] = np.empty((0, 0), dtype=FloatVal)
                else:
                    print(
                        f"Warning: Shape issue for {container_key_str}. Parsed {parsed_rows}x{parsed_cols}, "
                        f"got array shape {np_matrix.shape}. Using inferred array.")
                    final_container_typed['data'] = np_matrix
            except Exception as e:
                print(
                    f"Warning: Could not convert {container_key_str} data to NumPy array: {e}. "
                    f"Storing empty array with parsed dims.")
                final_container_typed['data'] = np.empty((parsed_rows, parsed_cols), dtype=FloatVal)
        else:  # list_of_rows is empty
            final_container_typed['data'] = np.empty((parsed_rows, parsed_cols), dtype=FloatVal)

        final_container_typed['rows'] = parsed_rows
        final_container_typed['cols'] = parsed_cols

    final_dat_data['dat_segment_count'] = parsed_segment_count
    final_dat_data['dat_segment_binary_filename'] = parsed_segment_binary_filename

    return final_dat_data


def load_segments_from_binary(
        bin_filepath: str,
        expected_segment_count_from_dat: Optional[int] = None
) -> Optional[List[Segment]]:
    """
    Loads segments from the specified binary file.
    Returns a list of NumPy arrays (segments), or None if loading fails.
    """
    segments_list: List[Segment] = []
    if not os.path.exists(bin_filepath):
        print(f"Error: Binary segment file not found: {bin_filepath}")
        return None

    try:
        with open(bin_filepath, 'rb') as bf:
            header_format: str = '<IHBI'
            header_size: int = struct.calcsize(header_format)
            header_bytes: bytes = bf.read(header_size)
            if len(header_bytes) < header_size:
                raise IOError(f"Binary file {bin_filepath} is too short to contain a valid header.")

            magic_number, format_version, element_type_id, bin_num_segments = struct.unpack(header_format, header_bytes)

            expected_magic: int = 0x5345474D
            if magic_number != expected_magic:
                raise ValueError(f"Invalid magic number. Expected {expected_magic:#x}, got {magic_number:#x}")

            if expected_segment_count_from_dat is not None and expected_segment_count_from_dat != bin_num_segments:
                print(
                    f"Warning: Segment count mismatch! .dat: {expected_segment_count_from_dat}, "
                    f".bin: {bin_num_segments}. Using .bin count.")

            if bin_num_segments == 0:
                print("Info: Binary file header indicates 0 segments.")
                return []

            np_dtype_val: Union[np.dtype[np.float32], np.dtype[np.float64]]
            if element_type_id == 0:
                np_dtype_val = np.dtype(np.float32)
            elif element_type_id == 1:
                np_dtype_val = np.dtype(np.float64)
            else:
                raise ValueError(f"Unknown element_type_id: {element_type_id}")

            segment_dims_list: List[Tuple[int, int, int]] = []
            descriptor_format: str = '<III'
            descriptor_size: int = struct.calcsize(descriptor_format)
            for i in range(bin_num_segments):
                descriptor_bytes = bf.read(descriptor_size)
                if len(descriptor_bytes) < descriptor_size:
                    raise IOError(
                        f"Binary file ended prematurely reading descriptor for segment {i + 1}/{bin_num_segments}.")
                dims = cast(Tuple[int, int, int], struct.unpack(descriptor_format, descriptor_bytes))
                segment_dims_list.append(dims)

            for i in range(bin_num_segments):
                dims = segment_dims_list[i]
                num_elements: int = int(np.prod(dims))

                current_segment: Segment
                if num_elements == 0:
                    current_segment = np.array([], dtype=np_dtype_val).reshape(dims)
                else:
                    bytes_to_read: int = num_elements * np_dtype_val.itemsize
                    segment_data_flat_bytes = bf.read(bytes_to_read)
                    if len(segment_data_flat_bytes) < bytes_to_read:
                        raise IOError(
                            f"Segment {i}: Expected {bytes_to_read} bytes, read {len(segment_data_flat_bytes)}.")

                    segment_data_flat = np.frombuffer(segment_data_flat_bytes, dtype=np_dtype_val, count=num_elements)

                    if segment_data_flat.size != num_elements:
                        raise IOError(
                            f"Segment {i}: After frombuffer, expected {num_elements} elements, "
                            f"got {segment_data_flat.size}")
                    try:
                        current_segment = segment_data_flat.reshape(dims)
                    except ValueError as reshape_error:
                        print(
                            f"Error reshaping segment {i}: {reshape_error}. Dims: {dims}, "
                            f"Elements: {segment_data_flat.size}")
                        return None
                segments_list.append(current_segment)
    except Exception as e:
        print(f"Error processing binary segment file {bin_filepath}: {e}")
        import traceback;
        traceback.print_exc()
        return None

    return segments_list


def load_full_segmentation_data(dat_filepath: str) -> Optional[FullSegmentationData]:
    """
    Orchestrates parsing of the .dat file and then the associated .bin segment file.
    Returns a single dictionary adhering to FullSegmentationData, or None on critical failure.
    """
    base_dir: str = os.path.dirname(os.path.abspath(dat_filepath))

    dat_info: Optional[DatFileData] = parse_dat_file(dat_filepath)
    if dat_info is None:
        print(f"Critical error: Failed to parse .dat file: {dat_filepath}")
        return None

    # Make a mutable copy for FullSegmentationData and ensure all keys from DatFileData are present.
    # Then add 'segment_container'.
    # The cast to FullSegmentationData assumes that all keys from DatFileData are present in dat_info.
    full_data: FullSegmentationData = {
        'SCALARS': dat_info.get('SCALARS', {}),
        'ORIGIN_CONTAINER': dat_info.get('ORIGIN_CONTAINER',
                                         cast(MatrixContainer, {'data': np.array([]), 'rows': 0, 'cols': 0})),
        'FACE_TO_GRID_INDEX_CONTAINER': dat_info.get('FACE_TO_GRID_INDEX_CONTAINER', cast(MatrixContainer,
                                                                                          {'data': np.array([]),
                                                                                           'rows': 0, 'cols': 0})),
        'FACE_TYPE_MAP': dat_info.get('FACE_TYPE_MAP', {}),
        'TYPE_COUNT_MAP': dat_info.get('TYPE_COUNT_MAP', {}),
        'VERT_TYPE_MAP': dat_info.get('VERT_TYPE_MAP', {}),
        'dat_segment_count': dat_info.get('dat_segment_count'),
        'dat_segment_binary_filename': dat_info.get('dat_segment_binary_filename'),
        'segment_container': []  # Initialize segment_container
    }

    dat_segment_count: Optional[int] = dat_info.get('dat_segment_count')
    relative_bin_filename: Optional[str] = dat_info.get('dat_segment_binary_filename')

    if dat_segment_count is not None and dat_segment_count > 0:
        if relative_bin_filename:
            bin_filepath: str = os.path.join(base_dir, relative_bin_filename)
            print(f"Attempting to load segments from: {bin_filepath}")
            segments: Optional[List[Segment]] = load_segments_from_binary(bin_filepath, dat_segment_count)
            if segments is not None:
                full_data['segment_container'] = segments
            else:
                print(f"Warning: Failed to load segments from {bin_filepath}. Segment container remains empty.")
        else:
            print(
                f"Warning: Segment count ({dat_segment_count}) > 0 in .dat, "
                f"but binary filename missing. Segments not loaded.")
    elif dat_segment_count == 0:
        print("Info: .dat file indicates 0 segments. Segment container is empty.")
    else:  # dat_segment_count is None
        print("Info: Segment count not specified or invalid in .dat. Segments not loaded from binary.")

    return full_data


def main():
    # Example usage:
    # Replace with your actual .dat file path
    dat_file_to_test = r'C:\Local_Data\ABC\ABC_statistics\benchmarks\ABC_chunk_benchmark\00000002\segmentation_data.dat'

    if os.path.exists(dat_file_to_test):
        print(f"\n--- Parsing actual file: {dat_file_to_test} ---")
        all_data = load_full_segmentation_data(dat_file_to_test)

        if all_data:
            print("\n--- Parsed Data from Actual File ---")
            if all_data.get('SCALARS'):
                print("Scalars:", all_data['SCALARS'])

            origin_cont = all_data.get('ORIGIN_CONTAINER')
            if origin_cont and isinstance(origin_cont.get('data'), np.ndarray):
                print(f"Origin Container (rows {origin_cont.get('rows', 0)}, cols {origin_cont.get('cols', 0)}), "
                      f"shape {origin_cont['data'].shape}")

            segments = all_data.get('segment_container', [])
            if segments:
                print(f"\nSegments Loaded: {len(segments)}")
                for i, seg_array in enumerate(segments):
                    if seg_array is not None: print(f"  Segment {i} shape: {seg_array.shape}, dtype: {seg_array.dtype}")
            else:
                print("\nNo segments loaded or segment_container is empty.")
        else:
            print("\nFull parsing of actual file failed.")
    else:
        print(f"Test .dat file not found: {dat_file_to_test}. Skipping actual file test.")
    print()


if __name__ == '__main__':
    main()