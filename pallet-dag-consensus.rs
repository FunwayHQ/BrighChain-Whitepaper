#![cfg_attr(not(feature = "std"), no_std)]

//! # DAG Consensus Pallet
//! 
//! This pallet implements the core DAG consensus rules and state transition logic
//! for the BrightChain network, providing a parallel transaction processing structure
//! based on the PHANTOM protocol with modifications for enhanced security and
//! deterministic finality.

pub use pallet::*;

#[frame_support::pallet]
pub mod pallet {
    use frame_support::{
        dispatch::DispatchResultWithPostInfo,
        pallet_prelude::*,
        traits::{Get, StorageVersion},
        weights::Weight,
    };
    use frame_system::pallet_prelude::*;
    use sp_runtime::{
        traits::{Hash, BlakeTwo256, Zero, One, CheckedAdd, SaturatedConversion},
        generic::{DigestItem, Era},
        Percent, Permill,
    };
    use sp_std::{prelude::*, collections::btree_map::BTreeMap, vec::Vec};
    use scale_info::TypeInfo;
    use codec::{Encode, Decode, MaxEncodedLen};

    /// The current storage version.
    const STORAGE_VERSION: StorageVersion = StorageVersion::new(1);

    /// A hash of a block.
    pub type BlockHash = sp_core::H256;

    /// Reference to parent blocks in the DAG structure
    #[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
    pub struct ParentReference {
        /// Hash of the parent block
        pub hash: BlockHash,
        /// Height of the parent block
        pub height: BlockNumber,
    }

    /// Block DAG metadata
    #[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
    pub struct BlockDagMeta<AccountId> {
        /// Block author/validator
        pub author: AccountId,
        /// Block creation timestamp
        pub timestamp: u64,
        /// References to parent blocks (typically 2-4)
        pub parents: Vec<ParentReference>,
        /// Accumulated block weight for GHOST fork choice
        pub cumulative_weight: u64,
        /// Block height range (min, max)
        pub height_range: (BlockNumber, BlockNumber),
        /// Flag if block is part of canonical chain
        pub is_canonical: bool,
    }

    /// GHOST-based metric for determining block weight
    #[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
    pub struct BlockWeight {
        /// Direct weight from validator stake
        pub stake_weight: u64,
        /// Accumulated descendant weight for GHOST calculation
        pub descendant_weight: u64,
    }

    /// Transaction conflict category
    #[derive(Clone, Copy, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
    pub enum ConflictType {
        /// No conflict detected
        None,
        /// Read-write conflict with another transaction
        ReadWrite,
        /// Write-write conflict with another transaction
        WriteWrite,
    }

    /// Transaction dependency information
    #[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
    pub struct TxDependency {
        /// Transaction hash this depends on
        pub tx_hash: [u8; 32],
        /// Type of conflict if executed out of order
        pub conflict_type: ConflictType,
    }

    #[pallet::pallet]
    #[pallet::storage_version(STORAGE_VERSION)]
    pub struct Pallet<T>(_);

    #[pallet::config]
    pub trait Config: frame_system::Config {
        /// The overarching event type.
        type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
        
        /// The maximum number of parent references a block can have
        #[pallet::constant]
        type MaxParents: Get<u32>;
        
        /// The maximum number of blocks to keep in DAG history
        #[pallet::constant]
        type MaxBlockHistory: Get<u32>;
        
        /// Weight information for extrinsics in this pallet
        type WeightInfo: WeightInfo;
        
        /// Minimum time between blocks from same validator
        #[pallet::constant]
        type MinBlockTime: Get<u64>;
    }

    /// Current blocks in the DAG, accessible by their hash
    #[pallet::storage]
    pub type BlockDag<T: Config> = StorageMap<
        _,
        Blake2_128Concat,
        BlockHash,
        BlockDagMeta<T::AccountId>,
        OptionQuery,
    >;

    /// All transactions in the DAG mapped by their hash
    #[pallet::storage]
    pub type Transactions<T: Config> = StorageMap<
        _,
        Blake2_128Concat,
        [u8; 32], // tx hash
        (BlockHash, u32), // (block hash, index)
        OptionQuery,
    >;

    /// Block height to hash mapping for canonical chain
    #[pallet::storage]
    pub type CanonicalChain<T: Config> = StorageMap<
        _,
        Blake2_128Concat,
        BlockNumber,
        BlockHash,
        OptionQuery,
    >;

    /// Current tips (blocks without children) in the DAG
    #[pallet::storage]
    pub type BlockTips<T: Config> = StorageValue<
        _,
        Vec<BlockHash>,
        ValueQuery,
    >;

    /// Calculated weights for GHOST fork choice rule
    #[pallet::storage]
    pub type BlockWeights<T: Config> = StorageMap<
        _,
        Blake2_128Concat,
        BlockHash,
        BlockWeight,
        OptionQuery,
    >;

    /// Transaction dependency graph for conflict detection
    #[pallet::storage]
    pub type TxDependencies<T: Config> = StorageMap<
        _,
        Blake2_128Concat,
        [u8; 32], // tx hash
        Vec<TxDependency>,
        ValueQuery,
    >;

    /// Last block production time per validator
    #[pallet::storage]
    pub type LastBlockTime<T: Config> = StorageMap<
        _,
        Blake2_128Concat,
        T::AccountId,
        u64,
        OptionQuery,
    >;

    /// Events generated by this pallet
    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        /// New block added to DAG [block_hash, author, parent_count]
        BlockAdded(BlockHash, T::AccountId, u32),
        /// Block became part of canonical chain [block_hash, height]
        BlockCanonical(BlockHash, BlockNumber),
        /// New tips in the DAG [tip_count]
        TipsUpdated(u32),
        /// Blocks pruned from DAG history [block_count]
        BlocksPruned(u32),
    }

    /// Errors for the DAG consensus pallet
    #[pallet::error]
    pub enum Error<T> {
        /// The block already exists in the DAG
        BlockExists,
        /// One or more parent blocks not found
        ParentNotFound,
        /// Too many parents specified
        TooManyParents,
        /// Block timestamp is invalid
        InvalidTimestamp,
        /// Block was authored too soon after author's previous block
        BlockRateLimited,
        /// Block height calculation failed
        InvalidHeight,
        /// Block conflicts with canonical chain
        ConflictsWithCanonical,
        /// Transaction already exists
        TransactionExists,
        /// Invalid transaction dependency
        InvalidDependency,
    }

    /// Extrinsic calls for the DAG consensus pallet
    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Submit a new block to the DAG
        #[pallet::weight(T::WeightInfo::submit_block())]
        pub fn submit_block(
            origin: OriginFor<T>,
            parent_hashes: Vec<BlockHash>,
            timestamp: u64,
            transactions: Vec<Vec<u8>>,
            weight: u64,
        ) -> DispatchResultWithPostInfo {
            let author = ensure_signed(origin)?;
            
            // Basic validation
            ensure!(
                parent_hashes.len() <= T::MaxParents::get() as usize,
                Error::<T>::TooManyParents
            );
            
            ensure!(
                timestamp <= Self::now(),
                Error::<T>::InvalidTimestamp
            );
            
            // Rate limiting check
            if let Some(last_time) = LastBlockTime::<T>::get(&author) {
                ensure!(
                    timestamp >= last_time + T::MinBlockTime::get(),
                    Error::<T>::BlockRateLimited
                );
            }
            
            // Create block hash - in a real implementation this would be based on
            // the merkle roots and other block data
            let block_data = (parent_hashes.clone(), timestamp, &author, &transactions);
            let block_hash = T::Hashing::hash_of(&block_data);
            
            // Ensure block doesn't already exist
            ensure!(!BlockDag::<T>::contains_key(&block_hash), Error::<T>::BlockExists);
            
            // Process parent references
            let parents: Vec<ParentReference> = parent_hashes
                .iter()
                .map(|hash| -> Result<ParentReference, Error<T>> {
                    let parent = BlockDag::<T>::get(hash).ok_or(Error::<T>::ParentNotFound)?;
                    Ok(ParentReference {
                        hash: *hash,
                        height: parent.height_range.1,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            
            // Calculate height range
            let min_height = parents
                .iter()
                .map(|p| p.height)
                .min()
                .unwrap_or(Zero::zero())
                .saturating_add(One::one());
                
            let max_height = parents
                .iter()
                .map(|p| p.height)
                .max()
                .unwrap_or(Zero::zero())
                .saturating_add(One::one());
                
            // Create block metadata
            let block_meta = BlockDagMeta {
                author: author.clone(),
                timestamp,
                parents: parents.clone(),
                cumulative_weight: weight,
                height_range: (min_height, max_height),
                is_canonical: false, // Will be updated by fork choice rule
            };
            
            // Store block in DAG and record last block time
            BlockDag::<T>::insert(&block_hash, block_meta);
            LastBlockTime::<T>::insert(&author, timestamp);
            
            // Update block tips - remove parents and add new block
            let mut tips = BlockTips::<T>::get();
            for parent in &parent_hashes {
                if let Some(pos) = tips.iter().position(|h| h == parent) {
                    tips.remove(pos);
                }
            }
            tips.push(block_hash);
            BlockTips::<T>::put(tips.clone());
            
            // Process transactions and update weights
            Self::process_transactions(&block_hash, &transactions)?;
            Self::update_ghost_weights(&block_hash, weight)?;
            Self::update_canonical_chain()?;
            
            // Prune old blocks if needed
            Self::prune_blocks();
            
            Self::deposit_event(Event::BlockAdded(block_hash, author, parents.len() as u32));
            Self::deposit_event(Event::TipsUpdated(tips.len() as u32));
            
            Ok(().into())
        }
        
        /// Explicitly trigger canonical chain recalculation
        /// This might be needed after network partitions or other issues
        #[pallet::weight(T::WeightInfo::recalculate_canonical_chain())]
        pub fn recalculate_canonical_chain(
            origin: OriginFor<T>
        ) -> DispatchResultWithPostInfo {
            ensure_root(origin)?;
            Self::update_canonical_chain()?;
            Ok(().into())
        }
    }

    // Internal helper functions
    impl<T: Config> Pallet<T> {
        /// Get current timestamp
        fn now() -> u64 {
            sp_io::timestamp::now().saturated_into::<u64>()
        }
        
        /// Process transactions in a block and update dependencies
        fn process_transactions(
            block_hash: &BlockHash,
            transactions: &[Vec<u8>],
        ) -> Result<(), Error<T>> {
            for (idx, tx_data) in transactions.iter().enumerate() {
                let tx_hash = BlakeTwo256::hash(tx_data);
                let tx_hash_bytes: [u8; 32] = tx_hash.into();
                
                // Ensure transaction doesn't already exist
                ensure!(
                    !Transactions::<T>::contains_key(tx_hash_bytes),
                    Error::<T>::TransactionExists
                );
                
                // Store transaction
                Transactions::<T>::insert(tx_hash_bytes, (*block_hash, idx as u32));
                
                // Here we would detect transaction dependencies based on state access
                // For this implementation, we'll just use a placeholder
                // In a real implementation, this would analyze read/write sets
                Self::detect_dependencies(&tx_hash_bytes, tx_data)?;
            }
            
            Ok(())
        }
        
        /// Detect and record transaction dependencies
        /// In a real implementation, this would analyze the transaction's read/write sets
        fn detect_dependencies(
            tx_hash: &[u8; 32],
            tx_data: &[u8],
        ) -> Result<(), Error<T>> {
            // Placeholder implementation
            let dependencies = Vec::new();
            TxDependencies::<T>::insert(tx_hash, dependencies);
            Ok(())
        }
        
        /// Update GHOST weights for fork choice rule
        fn update_ghost_weights(
            block_hash: &BlockHash,
            weight: u64,
        ) -> Result<(), Error<T>> {
            // Create initial block weight entry
            let block_weight = BlockWeight {
                stake_weight: weight,
                descendant_weight: 0,
            };
            BlockWeights::<T>::insert(block_hash, block_weight);
            
            // Get block's parents to update their descendant weights
            if let Some(block) = BlockDag::<T>::get(block_hash) {
                for parent in block.parents {
                    Self::update_ancestor_weights(&parent.hash, weight)?;
                }
            }
            
            Ok(())
        }
        
        /// Recursively update ancestor weights
        fn update_ancestor_weights(
            block_hash: &BlockHash,
            weight: u64,
        ) -> Result<(), Error<T>> {
            if let Some(mut block_weight) = BlockWeights::<T>::get(block_hash) {
                // Update descendant weight
                block_weight.descendant_weight = block_weight.descendant_weight.saturating_add(weight);
                BlockWeights::<T>::insert(block_hash, block_weight);
                
                // Recursively update ancestors
                if let Some(block) = BlockDag::<T>::get(block_hash) {
                    for parent in block.parents {
                         Self::update_ancestor_weights(&parent.hash, weight)?;
                    }
                }
            }
            
            Ok(())
        
        }
        
        /// Update the canonical chain based on GHOST fork choice rule
        fn update_canonical_chain() -> Result<(), Error<T>> {
            // Get all tips in the DAG
            let tips = BlockTips::<T>::get();
            if tips.is_empty() {
                return Ok(());
            }
            
            // Find the tip with highest total weight (stake + descendant)
            let mut best_tip = tips[0];
            let mut best_weight = if let Some(weight) = BlockWeights::<T>::get(&best_tip) {
                weight.stake_weight.saturating_add(weight.descendant_weight)
            } else {
                0
            };
            
            for tip in tips.iter().skip(1) {
                if let Some(weight) = BlockWeights::<T>::get(tip) {
                    let total_weight = weight.stake_weight.saturating_add(weight.descendant_weight);
                    if total_weight > best_weight {
                        best_tip = *tip;
                        best_weight = total_weight;
                    }
                }
            }
            
            // Build canonical chain by following parent links from best tip
            let mut canonical_blocks = Vec::new();
            let mut current_hash = best_tip;
            let mut visited = BTreeMap::new();
            
            while let Some(block) = BlockDag::<T>::get(&current_hash) {
                canonical_blocks.push((current_hash, block.height_range.1));
                
                // Mark block as canonical
                let mut updated_block = block.clone();
                updated_block.is_canonical = true;
                BlockDag::<T>::insert(&current_hash, updated_block);
                
                // Choose parent with highest height as next block
                if block.parents.is_empty() {
                    break;
                }
                
                // Find parent with highest height that hasn't been visited
                let mut next_parent = None;
                let mut max_height = Zero::zero();
                
                for parent in block.parents {
                    if !visited.contains_key(&parent.hash) && parent.height > max_height {
                        next_parent = Some(parent.hash);
                        max_height = parent.height;
                    }
                }
                
                if let Some(parent_hash) = next_parent {
                    visited.insert(current_hash, true);
                    current_hash = parent_hash;
                } else {
                    // No unvisited parents with valid height
                    break;
                }
            }
            
            // Update canonical chain mapping
            // Sort blocks by height in descending order
            canonical_blocks.sort_by(|a, b| b.1.cmp(&a.1));
            
            // Insert into canonical chain mapping
            for (hash, height) in canonical_blocks {
                CanonicalChain::<T>::insert(height, hash);
                Self::deposit_event(Event::BlockCanonical(hash, height));
            }
            
            Ok(())
        }
        
        /// Prune old blocks from DAG history
        fn prune_blocks() {
            let max_history = T::MaxBlockHistory::get() as usize;
            let mut blocks: Vec<_> = BlockDag::<T>::iter()
                .map(|(hash, meta)| (hash, meta.height_range.1))
                .collect();
            
            // Sort blocks by height in ascending order
            blocks.sort_by_key(|&(_, height)| height);
            
            // Keep only the most recent blocks
            if blocks.len() > max_history {
                let blocks_to_prune = blocks.len() - max_history;
                let pruned_blocks = blocks.drain(0..blocks_to_prune).collect::<Vec<_>>();
                
                for (hash, _) in pruned_blocks {
                    BlockDag::<T>::remove(hash);
                    BlockWeights::<T>::remove(hash);
                    
                    // We would also remove transactions here in a real implementation
                    // but we'll skip that for simplicity
                }
                
                Self::deposit_event(Event::BlocksPruned(blocks_to_prune as u32));
            }
        }
    }

    #[pallet::genesis_config]
    pub struct GenesisConfig<T: Config> {
        pub initial_authorities: Vec<T::AccountId>,
    }

    #[cfg(feature = "std")]
    impl<T: Config> Default for GenesisConfig<T> {
        fn default() -> Self {
            Self {
                initial_authorities: Vec::new(),
            }
        }
    }

    #[pallet::genesis_build]
    impl<T: Config> GenesisBuild<T> for GenesisConfig<T> {
        fn build(&self) {
            // Initialize an empty DAG structure
            BlockTips::<T>::put(Vec::<BlockHash>::new());
        }
    }
}

/// Weight information for extrinsics in this pallet
pub trait WeightInfo {
    fn submit_block() -> Weight;
    fn recalculate_canonical_chain() -> Weight;
}

/// Default weights implementation
impl WeightInfo for () {
    fn submit_block() -> Weight {
        Weight::from_parts(10_000, 0)
            .saturating_add(Weight::from_parts(0, 1024))
    }
    
    fn recalculate_canonical_chain() -> Weight {
        Weight::from_parts(10_000, 0)
            .saturating_add(Weight::from_parts(0, 1024))
    }
}