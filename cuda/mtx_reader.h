#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace mtx_reader {

struct MtxData {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    bool symmetric = false;
    std::vector<int> row_indices;
    std::vector<int> col_indices;
};

struct CsrGraph {
    int num_rows = 0;
    int num_cols = 0;
    int num_edges = 0;
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
};

inline std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

inline bool read_mtx_file(const std::string& path,
                          MtxData* out,
                          std::string* error,
                          bool expand_symmetric = true) {
    if (!out) {
        if (error) {
            *error = "Null output pointer";
        }
        return false;
    }

    std::ifstream file(path);
    if (!file) {
        if (error) {
            *error = "Failed to open file: " + path;
        }
        return false;
    }

    std::string line;
    if (!std::getline(file, line)) {
        if (error) {
            *error = "Empty file: " + path;
        }
        return false;
    }

    std::istringstream header_stream(line);
    std::string banner;
    std::string object;
    std::string format;
    std::string field;
    std::string symmetry;
    header_stream >> banner >> object >> format >> field >> symmetry;
    if (banner != "%%MatrixMarket") {
        if (error) {
            *error = "Missing MatrixMarket banner";
        }
        return false;
    }

    object = to_lower(object);
    format = to_lower(format);
    field = to_lower(field);
    symmetry = to_lower(symmetry);

    if (object != "matrix" || format != "coordinate") {
        if (error) {
            *error = "Only coordinate MatrixMarket files are supported";
        }
        return false;
    }

    bool is_symmetric = (symmetry == "symmetric" || symmetry == "skew-symmetric" ||
                         symmetry == "hermitian");

    int rows = 0;
    int cols = 0;
    int nnz = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') {
            continue;
        }
        std::istringstream size_stream(line);
        if (!(size_stream >> rows >> cols >> nnz)) {
            if (error) {
                *error = "Failed to read matrix size line";
            }
            return false;
        }
        break;
    }

    if (rows <= 0 || cols <= 0 || nnz < 0) {
        if (error) {
            *error = "Invalid matrix dimensions";
        }
        return false;
    }

    out->rows = rows;
    out->cols = cols;
    out->symmetric = is_symmetric;
    out->row_indices.clear();
    out->col_indices.clear();
    int reserve_nnz = nnz;
    if (is_symmetric && expand_symmetric) {
        reserve_nnz = nnz * 2;
    }
    out->row_indices.reserve(reserve_nnz);
    out->col_indices.reserve(reserve_nnz);

    int entries_read = 0;
    while (entries_read < nnz && std::getline(file, line)) {
        if (line.empty() || line[0] == '%') {
            continue;
        }
        std::istringstream entry_stream(line);
        int row = 0;
        int col = 0;
        if (!(entry_stream >> row >> col)) {
            if (error) {
                *error = "Failed to parse entry line";
            }
            return false;
        }
        --row;
        --col;
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            if (error) {
                *error = "Entry index out of bounds";
            }
            return false;
        }

        out->row_indices.push_back(row);
        out->col_indices.push_back(col);
        if (is_symmetric && expand_symmetric && row != col) {
            out->row_indices.push_back(col);
            out->col_indices.push_back(row);
        }
        ++entries_read;
    }

    if (entries_read != nnz) {
        if (error) {
            *error = "Unexpected end of file while reading entries";
        }
        return false;
    }

    out->nnz = static_cast<int>(out->row_indices.size());
    return true;
}

inline void build_csr(int num_rows,
                      int num_cols,
                      const std::vector<int>& row_indices,
                      const std::vector<int>& col_indices,
                      CsrGraph* out) {
    if (!out) {
        return;
    }
    out->num_rows = num_rows;
    out->num_cols = num_cols;
    out->num_edges = static_cast<int>(row_indices.size());
    out->row_offsets.assign(num_rows + 1, 0);
    out->col_indices.assign(out->num_edges, 0);

    for (int row : row_indices) {
        ++out->row_offsets[row + 1];
    }

    for (int row = 0; row < num_rows; ++row) {
        out->row_offsets[row + 1] += out->row_offsets[row];
    }

    std::vector<int> positions(out->row_offsets.begin(),
                               out->row_offsets.end() - 1);
    for (size_t i = 0; i < row_indices.size(); ++i) {
        int row = row_indices[i];
        int col = col_indices[i];
        int pos = positions[row]++;
        out->col_indices[pos] = col;
    }
}

inline bool read_mtx_as_csr(const std::string& path,
                            CsrGraph* out,
                            std::string* error,
                            bool expand_symmetric = true) {
    MtxData data;
    if (!read_mtx_file(path, &data, error, expand_symmetric)) {
        return false;
    }
    build_csr(data.rows, data.cols, data.row_indices, data.col_indices, out);
    return true;
}

}  // namespace mtx_reader
