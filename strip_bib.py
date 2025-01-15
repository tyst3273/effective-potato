
import re

# --------------------------------------------------------------------------------------------------

def get_citations_from_aux(aux_file):
    
    with open(aux_file,'r') as f:
        lines = f.readlines()

    citations = []
    for line in lines:
        if line.startswith('\citation'):
            line = line.strip('').split('{')[1].split('}')[0].split(',')
            for cite in line:
                if cite not in citations:
                    citations.append(cite)

    return citations

# --------------------------------------------------------------------------------------------------

def get_entry_from_bib(citation,bib_file):

    with open(bib_file,'r') as f:
        lines = f.readlines()

    for ii, line in enumerate(lines):

        if line.strip().startswith('@'):

            if re.search(citation,line) is None:
                continue

            count = 1
            jj = ii+1
            while True:

                line = lines[jj]
                num_open = len(re.findall('{',line))
                num_close = len(re.findall('}',line))
                count += num_open-num_close
                jj += 1

                if count == 0:
                    break

            entry = ''.join(lines[ii:jj])
            
    return entry
    
# --------------------------------------------------------------------------------------------------

aux_file = 'output.aux'
orig_bib = 'ref.bib'
stripped_bib = 'stripped.bib'

citations = get_citations_from_aux(aux_file)

with open(stripped_bib,'w') as f:

    for citation in citations:

        print(citation)
        entry = get_entry_from_bib(citation,orig_bib)    
    
        f.write(entry+'\n')
