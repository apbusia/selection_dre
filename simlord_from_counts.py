import os
import argparse
import tempfile
import pysam
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from sklearn.model_selection import KFold


SEED = 7


def main(args):
    savefile = args.save_file
    
    df = pd.read_csv(args.data_file)
    df = df[[args.seq_column, args.pre_column, args.post_column]]
    df = df.rename(columns={args.pre_column: 'pre_count',
                            args.post_column: 'post_count',
                            args.seq_column: 'seq'})
    output_dfs = [None] * args.n_folds
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    
    # Divide the library sequences.
    for i, (train_idx, test_idx) in enumerate(kf.split(df)):
        if savefile is not None:
            np.save('models/' + os.path.splitext(os.path.split(savefile)[1])[0] + '_fold{}_test_idx.npy'.format(i), test_idx)
        
        print_freq = 1000
        for pre_or_post in ['pre', 'post']:
            print('\nGenerating simlord {}-selection reads for fold {}...'.format(pre_or_post, i))
            cur_idx = train_idx[df['{}_count'.format(pre_or_post)].iloc[train_idx] > 0]
            n_variants = len(cur_idx)
            cur_df = None
            for v, idx in enumerate(cur_idx):
                if not v % print_freq:
                    print('\n\tVariant {} of {}...'.format(v, n_variants))
                count = df['{}_count'.format(pre_or_post)].iloc[idx]
                seq = df['seq'].iloc[idx]
                fasta_file = tempfile.NamedTemporaryFile('w')
                seq_record = SeqIO.SeqRecord(Seq(seq), id='{}_fold{}'.format(idx, i), description=args.data_file)
                SeqIO.write(seq_record, fasta_file, 'fasta')
                fasta_file.seek(0)
                simlord_name = 'simlord_reads'
                # https://janakiev.com/blog/python-shell-commands/
                os.system('simlord --read-reference {} -n {} {} -mr {} -xn 0.01 2.5394497 5500'.format(
                    fasta_file.name, count, simlord_name, len(seq)))
                with pysam.AlignmentFile('{}.sam'.format(simlord_name), 'r') as samfile:
                    reads = [read.query_sequence for read in samfile.fetch()]
                for ext in ['sam', 'fastq']:
                    os.remove('{}.{}'.format(simlord_name, ext))
                translate = lambda s: str(Seq(str(s)).translate())
                reads = pd.DataFrame(pd.Series(reads).apply(translate), columns=['seq'])
                if cur_df is None:
                    cur_df = reads
                else:
                    cur_df = cur_df.append(reads)
            print('\n\tComputing read counts...')
            cur_df = cur_df.value_counts()
            cur_df = cur_df.reset_index()
            cur_df.columns = ['seq', '{}_count'.format(pre_or_post)]
            if output_dfs[i] is None:
                output_dfs[i] = cur_df
            else:
                output_dfs[i] = output_dfs[i].merge(cur_df, how='outer', on='seq').fillna(0)
            del cur_df
            
    # Save combined datasets to file.
    if savefile is not None:
        savename, ext = os.path.splitext(savefile)
        for i, df in enumerate(output_dfs):
            if args.shuffle:
                df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
            df.to_csv(savename + '_fold{}'.format(i) + ext, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='path to counts dataset of long reads', type=str)
    parser.add_argument('--pre_column', default='count_pre', help='column name in counts dataset', type=str)
    parser.add_argument('--post_column', default='count_post', help='column name in counts dataset', type=str)
    parser.add_argument('--seq_column', default='seq', help='column name in counts dataset', type=str)
    parser.add_argument("--n_folds", default=1, help="number of folds into which to split the (combined) dataset", type=int)
    parser.add_argument("--save_file", help="path to which to save combined dataset", type=str)
    parser.add_argument("--shuffle", help="whether to shuffle combined dataset(s) before saving", action='store_true')
    args = parser.parse_args()
    main(args)