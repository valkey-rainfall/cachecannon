//! High-performance sharded histograms with metriken integration.
//!
//! Provides [`ShardedHistogram`] — a per-thread-sharded latency histogram that
//! avoids cross-thread cache-line contention by giving each worker its own
//! `AtomicHistogram` shard. Reads merge all shards into a single `Histogram`
//! snapshot.
//!
//! This mirrors the design of [`crate::sharded_counter::CounterGroup`]: each
//! thread writes to its own shard (selected by the SHARD_ID thread-local set
//! via [`crate::sharded_counter::set_thread_shard`]), eliminating the per-
//! response atomic RMW contention that serializes all workers on shared cache
//! lines at high throughput.

use std::sync::OnceLock;

use histogram::Histogram;
use metriken::AtomicHistogram;

use crate::sharded_counter::shard_index;

const NUM_SHARDS: usize = 64;

/// A sharded histogram that distributes writes across per-thread shards.
///
/// Each shard is a full `AtomicHistogram` with the same configuration
/// (grouping_power, max_value_power). Writes go to the calling thread's shard
/// (uncontended), reads merge all initialized shards via histogram addition.
pub struct ShardedHistogram {
    shards: [OnceLock<AtomicHistogram>; NUM_SHARDS],
    grouping_power: u8,
    max_value_power: u8,
}

// Safety: AtomicHistogram is Send+Sync, OnceLock is Send+Sync
unsafe impl Send for ShardedHistogram {}
unsafe impl Sync for ShardedHistogram {}

impl ShardedHistogram {
    /// Create a new sharded histogram. Each shard will be lazily initialized
    /// with the given parameters on first write.
    #[allow(clippy::declare_interior_mutable_const)]
    pub const fn new(grouping_power: u8, max_value_power: u8) -> Self {
        const EMPTY: OnceLock<AtomicHistogram> = OnceLock::new();
        Self {
            shards: [EMPTY; NUM_SHARDS],
            grouping_power,
            max_value_power,
        }
    }

    /// Record a value into the calling thread's shard.
    ///
    /// The shard is lazily initialized on first use. Returns `Ok(())` on
    /// success, or the error from the underlying histogram if the value is
    /// out of range.
    #[inline]
    pub fn increment(&self, value: u64) -> Result<(), histogram::Error> {
        let idx = shard_index();
        let shard = self.shards[idx].get_or_init(|| {
            AtomicHistogram::new(self.grouping_power, self.max_value_power)
        });
        shard.increment(value)
    }

    /// Load a merged snapshot of all initialized shards.
    ///
    /// Returns `None` if no shard has been written to. The merge uses
    /// `wrapping_add` to combine per-shard snapshots into a single histogram.
    pub fn load(&self) -> Option<Histogram> {
        let mut merged: Option<Histogram> = None;

        for lock in &self.shards {
            if let Some(shard) = lock.get() {
                if let Some(snapshot) = shard.load() {
                    merged = Some(match merged {
                        Some(acc) => acc.wrapping_add(&snapshot).unwrap_or(acc),
                        None => snapshot,
                    });
                }
            }
        }

        merged
    }
}

impl metriken::Metric for ShardedHistogram {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn value(&self) -> Option<metriken::Value<'_>> {
        // Return as Value::Other so the registry can downcast to us
        Some(metriken::Value::Other(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    use crate::sharded_counter::set_thread_shard;

    #[test]
    fn single_thread_increment_and_load() {
        let hist = ShardedHistogram::new(7, 64);
        set_thread_shard(0);

        hist.increment(100).unwrap();
        hist.increment(200).unwrap();
        hist.increment(300).unwrap();

        let snapshot = hist.load().expect("should have data");
        // The histogram should have 3 samples
        let count: u64 = snapshot.into_iter().map(|b| b.count()).sum();
        assert_eq!(count, 3);
    }

    #[test]
    fn multi_thread_increments_merge_correctly() {
        // Use a static so it's accessible from spawned threads
        static HIST: ShardedHistogram = ShardedHistogram::new(7, 64);

        let num_threads = 8;
        let increments_per_thread = 1000;

        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                thread::spawn(move || {
                    set_thread_shard(tid);
                    for i in 0..increments_per_thread {
                        // Values 1..=1000 (avoid 0 which may not bucket well)
                        HIST.increment((i + 1) as u64).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let snapshot = HIST.load().expect("should have data");
        let total_count: u64 = snapshot.into_iter().map(|b| b.count()).sum();
        assert_eq!(total_count, (num_threads * increments_per_thread) as u64);
    }

    #[test]
    fn empty_histogram_returns_none() {
        let hist = ShardedHistogram::new(7, 64);
        assert!(hist.load().is_none());
    }

    #[test]
    fn metriken_trait_returns_other() {
        use metriken::Metric;

        static HIST: ShardedHistogram = ShardedHistogram::new(7, 64);
        set_thread_shard(0);
        HIST.increment(42).unwrap();

        let value = Metric::value(&HIST);
        assert!(matches!(value, Some(metriken::Value::Other(_))));

        // Verify we can downcast back
        if let Some(metriken::Value::Other(any)) = value {
            assert!(any.downcast_ref::<ShardedHistogram>().is_some());
        }
    }

    #[test]
    fn shards_are_independent() {
        static HIST: ShardedHistogram = ShardedHistogram::new(7, 64);

        let h1 = thread::spawn(|| {
            set_thread_shard(10);
            for _ in 0..500 {
                HIST.increment(100).unwrap();
            }
        });
        let h2 = thread::spawn(|| {
            set_thread_shard(20);
            for _ in 0..500 {
                HIST.increment(200).unwrap();
            }
        });

        h1.join().unwrap();
        h2.join().unwrap();

        // Both shards should contribute to the merged total
        let snapshot = HIST.load().expect("should have data");
        let total: u64 = snapshot.into_iter().map(|b| b.count()).sum();
        assert_eq!(total, 1000);
    }
}
