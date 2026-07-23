//! A packed bit-per-row mask.

use serde::{Deserialize, Serialize};

/// One bit per row, packed eight to a byte.
///
/// Used for `present` / `stats_valid` in a [`StatColumn`](super::StatColumn),
/// where a collection of 10 000 datasets costs 1.25 KB per mask rather than a
/// `Vec<bool>`'s 10 KB.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Bitmap {
    bits: Vec<u8>,
    len: usize,
}

impl Bitmap {
    /// A bitmap of `len` unset bits.
    pub fn zeros(len: usize) -> Self {
        Self {
            bits: vec![0u8; len.div_ceil(8)],
            len,
        }
    }

    /// Number of bits.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Appends one bit.
    pub fn push(&mut self, value: bool) {
        if self.len.is_multiple_of(8) {
            self.bits.push(0);
        }
        let index = self.len;
        self.len += 1;
        self.set(index, value);
    }

    /// Reads bit `index`; out of range reads as `false`.
    pub fn get(&self, index: usize) -> bool {
        if index >= self.len {
            return false;
        }
        self.bits[index / 8] & (1u8 << (index % 8)) != 0
    }

    /// Writes bit `index`. Out-of-range writes are ignored.
    pub fn set(&mut self, index: usize, value: bool) {
        if index >= self.len {
            return;
        }
        let mask = 1u8 << (index % 8);
        if value {
            self.bits[index / 8] |= mask;
        } else {
            self.bits[index / 8] &= !mask;
        }
    }

    /// How many bits are set. Test-only cross-check on `present_mask`.
    #[cfg(test)]
    pub fn count_set(&self) -> usize {
        self.bits.iter().map(|b| b.count_ones() as usize).sum()
    }

    /// Grows to `len` bits, filling with `false`. Never shrinks.
    pub(crate) fn resize(&mut self, len: usize) {
        while self.len < len {
            self.push(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_and_count() {
        let mut bits = Bitmap::zeros(0);
        for i in 0..20 {
            bits.push(i % 3 == 0);
        }
        assert_eq!(bits.len(), 20);
        assert_eq!(bits.count_set(), 7);
        assert!(bits.get(0) && bits.get(3) && !bits.get(1));

        bits.set(1, true);
        assert!(bits.get(1));
        bits.set(0, false);
        assert!(!bits.get(0));
        assert!(!bits.get(100), "out of range reads as unset");
    }

    /// `count_set` counts whole bytes at a time, so it must ignore the padding
    /// bits in the final partial byte.
    #[test]
    fn count_set_ignores_trailing_padding() {
        let mut bits = Bitmap::zeros(0);
        for _ in 0..3 {
            bits.push(true);
        }
        assert_eq!(bits.len(), 3);
        assert_eq!(bits.count_set(), 3);

        // Setting beyond the length must not create phantom bits.
        bits.set(7, true);
        assert_eq!(bits.count_set(), 3);
    }

    #[test]
    fn resize_pads_with_unset() {
        let mut bits = Bitmap::zeros(2);
        bits.set(1, true);
        bits.resize(5);
        assert_eq!(bits.len(), 5);
        assert_eq!(bits.count_set(), 1);
        assert!(bits.get(1));
    }
}
