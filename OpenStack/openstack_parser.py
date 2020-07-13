import sys
sys.path.append('../')
from logparser import Spell, Drain
from OpenStack import const

def spell_parser(input_dir, output_dir):
    params = {
        'log_file': 'openstack.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'regex': [
            r"(?<=\[instance: )[A-Za-z0-9-]+", #instance id
            r'((\d+\.){3}\d+,?)+', # ip address
            r'/.+?\s', # /file path
            r'[\d.]+' # numbers
            ],
        'tau': 0.9,
        'keep_para':False
        }

    parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=params['log_format'], tau=params['tau'], rex=params['regex'], keep_para=params["keep_para"])
    parser.parse(params['log_file'])


if __name__ == "__main__":
    input_dir = const.OUTPUT_DIR + const.PARSER + "_result/"
    output_dir = const.OUTPUT_DIR + const.PARSER + "_result/"  # The output directory of parsing results
    if const.PARSER == 'spell':
        spell_parser(input_dir, output_dir)
    print("log parsing is done")
