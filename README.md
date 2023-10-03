# time-sift — creating time-lagged time series data

---

This crate provides the `lag_matrix` and related functions to create time-lagged versions of time series similar
to MATLAB's `lagmatrix` for time series analysis.

Support for [ndarray](https://crates.io/crates/ndarray)'s `Array1` and `Array2` traits is available via the
`ndarray` crate feature.

## Examples

For singular time series:

```rust
use timelag::lag_matrix;

fn singular_series() {
    let data = [1.0, 2.0, 3.0, 4.0];

    // Using infinity for padding because NaN doesn't equal itself.
    let lag = f64::INFINITY;
    let padding = f64::INFINITY;

    // Create three lagged versions.
    // Use a stride of 5 for the rows, i.e. pad with one extra entry.
    let lagged = lag_matrix(&data, 0..=3, lag, 5).unwrap();

    assert_eq!(
        lagged,
        &[
            1.0, 2.0, 3.0, 4.0, padding, // original data
            lag, 1.0, 2.0, 3.0, padding, // first lag
            lag, lag, 1.0, 2.0, padding, // second lag
            lag, lag, lag, 1.0, padding, // third lag
        ]
    );

    // The function is also available via the CreateLagMatrix trait.
    // All methods take an IntoIterator<Item = usize> for the lags.
    // The stride can be defaulted to `0` to produce tightly-packed consecutive values.
    let lagged = data.lag_matrix([0, 1, 2, 3], lag, 0).unwrap();
    assert_eq!(
        lagged,
        &[
            1.0, 2.0, 3.0, 4.0,
            lag, 1.0, 2.0, 3.0,
            lag, lag, 1.0, 2.0,
            lag, lag, lag, 1.0,
        ]
    );

    assert_eq!(lagged, other);
}
```

For matrices with time series along their rows:

```rust
use timelag::{lag_matrix_2d, MatrixLayout};

fn matrix_rows() {
    let data = [
         1.0,  2.0,  3.0,  4.0,
        -1.0, -2.0, -3.0, -4.0
    ];

    // Using infinity for padding because NaN doesn't equal itself.
    let lag = f64::INFINITY;
    let padding = f64::INFINITY;

    let lagged = lag_matrix_2d(&data, MatrixLayout::RowWise(4), 0..=3, lag, 5).unwrap();

    assert_eq!(
        lagged,
        &[
             1.0,  2.0,  3.0,  4.0, padding, // original data
            -1.0, -2.0, -3.0, -4.0, padding,
             lag,  1.0,  2.0,  3.0, padding, // first lag
             lag, -1.0, -2.0, -3.0, padding,
             lag,  lag,  1.0,  2.0, padding, // second lag
             lag,  lag, -1.0, -2.0, padding,
             lag,  lag,  lag,  1.0, padding, // third lag
             lag,  lag,  lag, -1.0, padding,
        ]
    );
}
```

For matrices with time series along their columns:

```rust
use timelag::{lag_matrix_2d, MatrixLayout};

fn matrix_columns() {
    let data = [
        1.0, -1.0,
        2.0, -2.0,
        3.0, -3.0,
        4.0, -4.0
    ];

    // Using infinity for padding because NaN doesn't equal itself.
    let lag = f64::INFINITY;
    let padding = f64::INFINITY;

    // Example row stride of nine: 2 time series × (1 original + 3 lags) + 1 extra padding.
    let lagged = lag_matrix_2d(&data, MatrixLayout::ColumnWise(4), 0..=3, lag, 9).unwrap();

    assert_eq!(
        lagged,
        &[
        //   original
        //   |-----|    first lag
        //   |     |     |-----|    second lag
        //   |     |     |     |     |-----|    third lag
        //   |     |     |     |     |     |     |-----|
        //   ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓
            1.0, -1.0,  lag,  lag,  lag,  lag,  lag,  lag, padding,
            2.0, -2.0,  1.0, -1.0,  lag,  lag,  lag,  lag, padding,
            3.0, -3.0,  2.0, -2.0,  1.0, -1.0,  lag,  lag, padding,
            4.0, -4.0,  3.0, -3.0,  2.0, -2.0,  1.0, -1.0, padding
        ]
    );
}
```
