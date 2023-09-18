import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)


def verts_collate(batch):
    databatch = [b['inp'] for b in batch]
    
    lenbatch = [b.shape[-1] for b in databatch]
  


    databatchTensor = torch.stack(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'au' in batch[0]:
        aubatch = [b['au'] for b in batch]
        cond['y'].update({'au': torch.stack(aubatch)})

    if 'au_raw' in batch[0]:
        aurawbatch = [b['au_raw'] for b in batch]
        cond['y'].update({'au_raw': torch.stack(aurawbatch)})
        
    # Action label
    if 'action' in batch[0]:
        actionbatch = [b['action'] for b in batch]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    if 'ldmks' in batch[0]:
        ldmks_batch = [b['ldmks'] for b in batch]
        cond['y'].update({'ldmks': torch.stack(ldmks_batch)})

    # Facs translation
    if 'trans' in batch[0]:
        trans_batch = [b['trans'] for b in batch]
        cond['y'].update({'trans': torch.stack(trans_batch)})

    if 'var' in batch[0]:
        var_batch = [b['var'] for b in batch]
        cond['y'].update({'var': torch.as_tensor(var_batch).unsqueeze(1)})
    
    # collate action textual names
    if 'action_text' in batch[0]:
        action_text = [b['action_text']for b in batch]
        cond['y'].update({'action_text': action_text})


    return motion, cond

def facs_collate(batch):
    databatch = [b['inp'].unsqueeze(0) for b in batch]
    
    lenbatch = [b.shape[-1] for b in databatch]
  


    databatchTensor = torch.stack(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'ldmks' in batch[0]:
        ldmks_batch = [b['ldmks'] for b in batch]
        cond['y'].update({'ldmks': torch.stack(ldmks_batch)})

    if 'trans' in batch[0]:
        trans_batch = [b['trans'] for b in batch]
        cond['y'].update({'trans': torch.stack(trans_batch)})

    if 'action_text' in batch[0]:
        action_text = [b['action_text']for b in batch]
        cond['y'].update({'action_text': action_text})
    
    return motion, cond  
