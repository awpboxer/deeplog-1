import sys
sys.path.append('../')
from logparser import Spell, Drain
from HDFS import hdfs_const

def spell_parser(input_dir, output_dir, log_file, log_format):
    tau        = 0.55  # Message type threshold (default: 0.5)
    regex      = [
        "(/[-\w]+)+", #replace file path with *
        "(?<=blk_)[-\d]+" #replace block_id with *

    ]  # Regular expression list for optional preprocessing (default: [])

    parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
    parser.parse(log_file)


def drain_parser(input_dir, output_dir, log_file, log_format):
    # Regular expression list for optional preprocessing (default: [])
    regex = [
        r'blk_(|-)[0-9]+',  # block id
        r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
        r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
    ]
    # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
    st = 0.5  # Similarity threshold
    depth = 3  # Depth of all leaf nodes

    parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
    parser.parse(log_file)

if __name__ == "__main__":
    # parse HDFS log
    input_dir = '../../data/hdfs_loghub/HDFS_1/'  # The input directory of log file
    output_dir = '../output/hdfs/' + hdfs_const.PARSER + "_result/"  # The output directory of parsing results
    log_file = 'HDFS.log'  # The input log file name
    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
    if hdfs_const.PARSER == "spell":
        spell_parser(input_dir, output_dir, log_file, log_format)
    elif hdfs_const.PARSER == "drain":
        drain_parser(input_dir, output_dir, log_file, log_format)
