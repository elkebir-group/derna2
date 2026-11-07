use derna_codon_dp::codon_table::{
    aa_to_index, get_codon_index, index_to_aa, AminoAcidError, CodonError, AMINO_ACIDS, N_CODON,
    STOP, STOP_CODE,
};

#[test]
fn aa_index_first_and_last() {
    assert_eq!(aa_to_index('A').unwrap(), 0);
    assert_eq!(aa_to_index('V').unwrap(), 19);
}

#[test]
fn aa_index_all_twenty() {
    let expected = [
        ('A', 0),
        ('R', 1),
        ('N', 2),
        ('D', 3),
        ('C', 4),
        ('Q', 5),
        ('E', 6),
        ('G', 7),
        ('H', 8),
        ('M', 9),
        ('I', 10),
        ('L', 11),
        ('K', 12),
        ('F', 13),
        ('P', 14),
        ('S', 15),
        ('T', 16),
        ('W', 17),
        ('Y', 18),
        ('V', 19),
    ];
    for (code, idx) in expected {
        assert_eq!(aa_to_index(code).unwrap(), idx, "aa_to_index({code:?})");
        assert_eq!(index_to_aa(idx).unwrap(), code, "index_to_aa({idx})");
    }
}

#[test]
fn aa_index_unknown_returns_error() {
    assert_eq!(aa_to_index('X'), Err(AminoAcidError::UnknownCode('X')));
    assert_eq!(aa_to_index('B'), Err(AminoAcidError::UnknownCode('B')));
}

#[test]
fn index_to_aa_out_of_range_returns_error() {
    assert_eq!(index_to_aa(21), Err(AminoAcidError::IndexOutOfRange(21)));
}

#[test]
fn n_codon_counts_each_familys_synonyms() {
    // Synonym counts per family: A=4, R=6, N=2, D=2, C=2, Q=2, E=2, G=4,
    // H=2, M=1, I=3, L=6, K=2, F=2, P=4, S=6, T=4, W=1, Y=2, V=4; then the
    // stop family's three.
    assert_eq!(
        N_CODON,
        [4, 6, 2, 2, 2, 2, 2, 4, 2, 1, 3, 6, 2, 2, 4, 6, 4, 1, 2, 4, 3]
    );
}

#[test]
fn the_twenty_spell_the_sixty_one_sense_codons() {
    let sense: usize = AMINO_ACIDS[..STOP].iter().map(|aa| aa.codons.len()).sum();
    assert_eq!(sense, 61);
}

#[test]
fn the_stop_family_is_the_three_stop_codons_after_the_twenty() {
    assert_eq!(STOP, 20);
    let stop = &AMINO_ACIDS[STOP];
    assert_eq!((stop.code, stop.index), (STOP_CODE, STOP));
    let names: Vec<&str> = stop.codons.iter().map(|c| c.name).collect();
    assert_eq!(names, ["UAA", "UAG", "UGA"]);
    assert_eq!(index_to_aa(STOP), Ok('*'));
    // Each stop codon is a synonym of the stop family and of nothing else.
    for (ci, codon) in stop.codons.iter().enumerate() {
        assert_eq!(get_codon_index(STOP, codon.nuc), Ok(ci));
        for aa in 0..STOP {
            assert!(get_codon_index(aa, codon.nuc).is_err(), "{}", codon.name);
        }
    }
}

#[test]
fn a_protein_cannot_name_the_stop_family() {
    // A stop slot is asked for separately; `*` in a protein is refused.
    assert_eq!(aa_to_index('*'), Err(AminoAcidError::UnknownCode('*')));
}

#[test]
fn n_codon_matches_amino_acids_len() {
    for (i, aa) in AMINO_ACIDS.iter().enumerate() {
        assert_eq!(
            N_CODON[i],
            aa.codons.len(),
            "N_CODON[{i}] vs AMINO_ACIDS[{i}].codons.len()"
        );
    }
}

#[test]
fn get_codon_index_spot_checks() {
    // A → GCU is ci=0: G=2, C=1, U=3
    assert_eq!(get_codon_index(0, [2, 1, 3]).unwrap(), 0);
    // A → GCC is ci=1: G=2, C=1, C=1
    assert_eq!(get_codon_index(0, [2, 1, 1]).unwrap(), 1);
    // M → AUG is ci=0 (only codon): A=0, U=3, G=2
    assert_eq!(get_codon_index(9, [0, 3, 2]).unwrap(), 0);
    // L → CUG is ci=3: C=1, U=3, G=2
    assert_eq!(get_codon_index(11, [1, 3, 2]).unwrap(), 3);
    // S → AGC is ci=5: A=0, G=2, C=1
    assert_eq!(get_codon_index(15, [0, 2, 1]).unwrap(), 5);
}

#[test]
fn get_codon_index_rejects_mismatch() {
    // GGC is Gly (aa=7), not Ala (aa=0)
    assert_eq!(
        get_codon_index(0, [2, 2, 1]),
        Err(CodonError::NotASynonym {
            aa_idx: 0,
            codon: [2, 2, 1]
        })
    );
}

#[test]
fn codon_names_are_valid_rna() {
    for aa in &AMINO_ACIDS {
        for codon in aa.codons {
            assert_eq!(codon.name.len(), 3, "codon name length: {}", codon.name);
            for ch in codon.name.chars() {
                assert!(
                    matches!(ch, 'A' | 'C' | 'G' | 'U'),
                    "invalid nucleotide {ch:?} in codon {}",
                    codon.name
                );
            }
        }
    }
}

#[test]
fn codon_nuc_matches_name() {
    // Verify the nuc encoding matches A=0, C=1, G=2, U=3 for each codon name.
    fn encode(c: char) -> u8 {
        match c {
            'A' => 0,
            'C' => 1,
            'G' => 2,
            'U' => 3,
            _ => panic!("bad nuc {c}"),
        }
    }
    for aa in &AMINO_ACIDS {
        for codon in aa.codons {
            let expected: [u8; 3] = {
                let cs: Vec<char> = codon.name.chars().collect();
                [encode(cs[0]), encode(cs[1]), encode(cs[2])]
            };
            assert_eq!(codon.nuc, expected, "codon {} nuc mismatch", codon.name);
        }
    }
}
