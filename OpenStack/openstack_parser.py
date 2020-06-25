import sys
sys.path.append('../')
from logparser import Spell, Drain
from OpenStack import const

def spell_parser(input_dir, output_dir, log_file, log_format):
    tau        = 0.55  # Message type threshold (default: 0.5)
    regex      = [
        "(/[-\w]+)+", #replace file path with *
        "(?<=blk_)[-\d]+" #replace block_id with *

    ]  # Regular expression list for optional preprocessing (default: [])

    parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
    parser.has_list = True  # connect request ids in content with %
    parser.parse(log_file)



if __name__ == "__main__":
    input_dir = const.DATA_DIR
    output_dir = const.OUTPUT_DIR + const.PARSER + "_result/"  # The output directory of parsing results
    log_file = 'OpenStack_2k.log'  # The input log file name


    log_format = '<Filename> <Date> <Time> <Pid> <Level> <Component> <Request_id> <Content>'  # openstack log format
    tau = 0.55  # Message type threshold (default: 0.5)
    regex = [
        r"(?<=\[instance: )[A-Za-z0-9-]+" , #replace id after instance with *
        r"([A-Za-z0-9]+-){4}[A-Za-z0-9]+",  # replace instance id with *
        r"(/\w+)+"  # replace file address /var/.../.. with *
    ]
    # Regular expression list for optional preprocessing (default: []) for line in log

    parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
    parser.has_list = True
    parser.parse(log_file)
