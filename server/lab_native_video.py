"""Native video-file input through Qwen's official utilities; no evidence extraction."""
from pathlib import Path

SYSTEM='You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
# T4 SDPA cannot hold the default ~18k-token attention matrix. Keep Qwen's
# temporal sampling and original MP4 path, while bounding only pixels per frame.
VIDEO_MAX_PIXELS=160*28*28


def conversation_for(path, text, media_mode='av', max_pixels=VIDEO_MAX_PIXELS):
    if media_mode not in ('av','vision','audio'):raise ValueError('Unknown media mode')
    media=({'type':'audio','audio':str(path)} if media_mode=='audio' else
           {'type':'video','video':str(path),'max_pixels':max_pixels})
    return [
        {'role':'system','content':[{'type':'text','text':SYSTEM}]},
        {'role':'user','content':[media,
                                  {'type':'text','text':text}]},
    ]


def prepare_video(path, max_pixels=VIDEO_MAX_PIXELS):
    from qwen_omni_utils import process_mm_info
    path=Path(path).resolve()
    if not path.is_file():raise FileNotFoundError(path)
    # MP4 path, not caller-extracted frames. Keep Qwen's default temporal sampling;
    # max_pixels per frame only bounds resolution for the T4 memory budget.
    # The notebook selects Qwen's supported Decord reader because Kaggle's
    # torchvision build no longer provides torchvision.io.read_video.
    audios,images,videos,video_kwargs=process_mm_info(
        conversation_for(path,'',max_pixels=max_pixels),use_audio_in_video=True,return_video_kwargs=True)
    if images is not None:raise RuntimeError('Unexpected supplemental images')
    if audios is None or len(audios)!=1 or videos is None or len(videos)!=1:
        raise RuntimeError('Expected one original video and its audio')
    # Transformers 4.57.6 Omni expects a scalar fps for this single video.
    fps=video_kwargs['fps']
    if isinstance(fps,(list,tuple)):
        if len(fps)!=1:raise RuntimeError('Expected one video fps')
        fps=fps[0]
    return {'path':str(path),'audio':audios,'videos':videos,'fps':float(fps),
            'max_pixels_per_frame':max_pixels}


def qwen_inputs(processor, prompt, media, media_mode='av'):
    if media_mode not in ('av','vision','audio'):raise ValueError('Unknown media mode')
    use_audio=media_mode in ('av','audio')
    use_video=media_mode in ('av','vision')
    rendered=processor.apply_chat_template(conversation_for(media['path'],prompt,media_mode,media['max_pixels_per_frame']),
        add_generation_prompt=True,tokenize=False)
    inputs=processor(text=rendered,audio=media['audio'] if use_audio else None,images=None,videos=media['videos'] if use_video else None,
        return_tensors='pt',padding=True,
        videos_kwargs={'fps':media['fps'],'use_audio_in_video':media_mode=='av'})
    if ('input_features' in inputs)!=use_audio or ('pixel_values_videos' in inputs)!=use_video:
        raise RuntimeError('Native inputs do not match the requested modalities')
    if 'pixel_values' in inputs:raise RuntimeError('Unexpected supplemental image tokens')
    # HF Transformers takes this switch from videos_kwargs, not its top-level
    # kwargs. Check that the resulting token sequence actually contains audio.
    audio_token_id=processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
    audio_tokens=int((inputs['input_ids']==audio_token_id).sum().item())
    expected_audio_tokens=0
    if use_audio:
        if 'feature_attention_mask' not in inputs:
            raise RuntimeError('Audio feature mask missing; cannot verify audio-token alignment')
        input_lengths=(inputs['feature_attention_mask'].sum(-1)-1)//2+1
        expected_audio_tokens=int((((input_lengths-2)//2+1).clamp_min(0)).sum().item())
    if audio_tokens!=expected_audio_tokens or (use_audio and audio_tokens==0):
        raise RuntimeError(f'Audio tokens missing/misaligned: found {audio_tokens}, expected {expected_audio_tokens}')
    expected_video_tokens=0
    if use_video:
        grid=inputs['video_grid_thw']
        merge=processor.video_processor.merge_size**2
        expected_video_tokens=int((grid.prod(-1).sum()//merge).item())
    video_token_id=processor.tokenizer.convert_tokens_to_ids(processor.video_token)
    video_tokens=int((inputs['input_ids']==video_token_id).sum().item())
    if video_tokens!=expected_video_tokens or (use_video and video_tokens==0):
        raise RuntimeError(f'Video tokens missing/misaligned: found {video_tokens}, expected {expected_video_tokens}')
    return inputs,rendered


def move_inputs(inputs, model):
    import torch
    inputs=inputs.to(model.get_input_embeddings().weight.device)
    for key in ('input_features','pixel_values_videos'):
        if key in inputs:inputs[key]=inputs[key].to(torch.float16)
    return inputs
