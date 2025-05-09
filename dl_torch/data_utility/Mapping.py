def linear_mapping(min_val, max_val, min_map, max_map, value):
    m =  (max_map - min_map)/(max_val -min_val)
    n = max_map - (max_map - min_map)/(max_val -min_val) * max_val
    return m*value + n