
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import gc

# 导入你的模型 - 使用和训练相同的模型类
from models.model_re_bbox import Bench
from models.tokenization_bert import BertTokenizer
from models.tokenization_roberta import RobertaTokenizer
from ruamel.yaml import YAML

class GeoTextImageDataset(Dataset):
    """处理地理文本检索图像的数据集"""
    def __init__(self, image_dir, image_size=384):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        
        # 查找所有图像文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        self.image_files = []
        
        # 在当前目录查找
        for ext in image_extensions:
            self.image_files.extend(list(self.image_dir.glob(ext)))
            self.image_files.extend(list(self.image_dir.glob(ext.upper())))
        
        # 在子目录中递归查找
        for ext in image_extensions:
            self.image_files.extend(list(self.image_dir.rglob(ext)))
            self.image_files.extend(list(self.image_dir.rglob(ext.upper())))
        
        # 去重并排序
        self.image_files = sorted(list(set(self.image_files)))
        
        print(f"Found {len(self.image_files)} image files")
        
        if len(self.image_files) > 0:
            print("Sample files:")
            for i, f in enumerate(self.image_files[:5]):
                print(f"  {i+1}. {f.name}")
            if len(self.image_files) > 5:
                print(f"  ... and {len(self.image_files) - 5} more files")
        


        self.transform = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.48821073],
                        std=[0.26862954, 0.26130258, 0.27577711])  # 使用CLIP参数
])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            # 返回图像和文件名（不带扩展名，用于匹配）
            return image, image_path.stem
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # 返回黑色图像作为fallback
            black_image = Image.new('RGB', (self.image_size, self.image_size), color='black')
            if self.transform:
                black_image = self.transform(black_image)
            return black_image, image_path.stem

def parse_queries(query_file):
    """从txt文件中解析查询"""
    queries = []
    query_ids = []
    
    try:
        with open(query_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        original_qid, text = parts
                        # 转换查询ID格式：从 q_00001 转为 q1，或从其他格式转为 q数字
                        if original_qid.startswith('q_'):
                            # 提取数字部分并去掉前导零
                            number_part = original_qid[2:].lstrip('0') or '0'
                            new_qid = f"q{number_part}"
                        elif original_qid.startswith('q') and original_qid[1:].isdigit():
                            # 如果已经是 q1, q2 这样的格式，保持不变
                            new_qid = original_qid
                        else:
                            # 如果是纯数字，转换为 q数字
                            if original_qid.isdigit():
                                new_qid = f"q{original_qid}"
                            else:
                                new_qid = original_qid
                        
                        query_ids.append(new_qid)
                        queries.append(text)
                        
                        # 显示前几个转换示例
                        if line_num <= 3:
                            print(f"Query ID conversion: '{original_qid}' -> '{new_qid}'")
                    else:
                        print(f"Warning: Line {line_num} format incorrect, skipping: {line[:50]}...")
        
        print(f"Loaded {len(queries)} queries from {query_file}")
        return queries, query_ids
    
    except FileNotFoundError:
        print(f"Error: Query file '{query_file}' not found!")
        return [], []
    except Exception as e:
        print(f"Error reading query file: {e}")
        return [], []

def create_tokenizer(text_encoder_path, use_roberta=False):
    """创建分词器，匹配训练代码的方式"""
    print(f"Loading tokenizer from: {text_encoder_path}")
    
    # 检查本地路径是否存在
    if os.path.exists(text_encoder_path):
        print(f"✓ Local path exists: {text_encoder_path}")
        
        # 列出目录中的文件
        try:
            files_in_dir = os.listdir(text_encoder_path)
            print(f"Files in directory: {files_in_dir}")
        except Exception as e:
            print(f"Cannot list directory contents: {e}")
        
        # 检查必要的tokenizer文件
        vocab_file = os.path.join(text_encoder_path, 'vocab.txt')
        config_file = os.path.join(text_encoder_path, 'tokenizer_config.json')
        
        print(f"Checking vocab.txt: {os.path.exists(vocab_file)}")
        print(f"Checking tokenizer_config.json: {os.path.exists(config_file)}")
        
        # 方法1: 直接使用vocab.txt文件创建tokenizer（匹配训练代码）
        if os.path.exists(vocab_file):
            try:
                print("Method 1: Creating tokenizer directly from vocab file...")
                if use_roberta:
                    # RoBERTa需要额外的文件，先尝试标准方法
                    tokenizer = RobertaTokenizer.from_pretrained(text_encoder_path)
                else:
                    # 对于BERT，可以直接从vocab.txt创建
                    tokenizer = BertTokenizer.from_pretrained(text_encoder_path)
                print(f"✓ Successfully created tokenizer directly")
                return tokenizer
            except Exception as e:
                print(f"✗ Direct creation failed: {e}")
        
        # 方法2: 使用from_pretrained方法（允许在线回退）
        try:
            print("Method 2: Using from_pretrained with online fallback...")
            if use_roberta:
                tokenizer = RobertaTokenizer.from_pretrained(text_encoder_path)
            else:
                tokenizer = BertTokenizer.from_pretrained(text_encoder_path)
            print(f"✓ Successfully loaded tokenizer using from_pretrained")
            return tokenizer
        except Exception as e:
            print(f"✗ from_pretrained method failed: {e}")
    
    else:
        print(f"✗ Local path does not exist: {text_encoder_path}")
    
    # 最终fallback：使用标准预训练模型
    try:
        print("Final fallback: Using standard pretrained model...")
        if use_roberta:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            print("✓ Successfully loaded RoBERTa tokenizer from HuggingFace")
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            print("✓ Successfully loaded BERT tokenizer from HuggingFace")
        return tokenizer
    except Exception as e:
        print(f"✗ All tokenizer loading methods failed: {e}")
        raise RuntimeError("Failed to load tokenizer from any source")

def load_xvlm_model(checkpoint_path, config_path, device):

    print("Loading configuration...")
    yaml = YAML(typ='rt')
    with open(config_path, 'r') as f:
        config = yaml.load(f)
    
    print(f"Original config - image_res: {config.get('image_res', 'Not set')}, patch_size: {config.get('patch_size', 'Not set')}")
    

    # 使用和训练完全相同的模型创建方式
    model = Bench(config=config)
    
    # 计算并显示模型参数总量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Parameter Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"  Model size (approx): {total_params * 4 / 1024**2:.2f} MB (FP32)")
    print()
    
    # 显示各组件的参数分布
    print("📋 Parameter distribution by component:")
    component_params = {}
    for name, param in model.named_parameters():
        component = name.split('.')[0]  # 获取组件名称
        if component not in component_params:
            component_params[component] = 0
        component_params[component] += param.numel()
    
    for component, params in sorted(component_params.items()):
        percentage = params / total_params * 100
        print(f"  {component}: {params:,} ({percentage:.1f}%)")
    print()
    
    print("Loading checkpoint using model's load_pretrained method...")
    # 使用模型自己的加载方法，这样可以确保和训练时一致
    try:
        model.load_pretrained(checkpoint_path, config, is_eval=True)
    except Exception as e:
        print(f"Failed to use model's load_pretrained method: {e}")
        print("Falling back to manual loading...")
        
        # 手动加载作为备选
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # 不过滤任何参数，尝试加载所有内容
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Manual loading - Missing keys: {len(msg.missing_keys)}")
        print(f"Manual loading - Unexpected keys: {len(msg.unexpected_keys)}")
    
    model = model.to(device)
    model.eval()
    model = model.half()  # 使用FP16以匹配训练时的evaluation
    
    # 禁用梯度计算
    for param in model.parameters():
        param.requires_grad = False
    
    # 重新计算参数统计（加载后）
    total_params_after = sum(p.numel() for p in model.parameters())
    trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Statistics After Loading:")
    print(f"  Model training mode: {model.training}")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Data type: {next(model.parameters()).dtype}")
    print(f"  Total parameters: {total_params_after:,}")
    print(f"  Trainable parameters: {trainable_params_after:,}")
    print(f"  Model size (FP16): {total_params_after * 2 / 1024**2:.2f} MB")
    print()
    
    # 创建分词器 - 匹配训练代码的方式
    print("Creating tokenizer...")
    text_encoder_path = config['text_encoder']
    use_roberta = config.get('use_roberta', False)
    
    tokenizer = create_tokenizer(text_encoder_path, use_roberta)
    
    return model, tokenizer, config

def extract_all_features(model, tokenizer, queries, image_loader, config, device, add_prefix=False):
    """提取所有特征 - 完全模仿训练代码的evaluation方法"""
    print("Extracting all features using training code's evaluation approach...")
    
    # ===== 提取文本特征 =====
    print("Extracting text features...")
    texts = queries
    num_text = len(texts)
    text_bs = config.get('batch_size_test_text', 256)  # 使用和训练相同的batch size
    text_feats = []
    text_embeds = []  
    text_atts = []
    
    with torch.no_grad():
        for i in range(0, num_text, text_bs):
            text = texts[i: min(num_text, i + text_bs)]
            
            # 如果需要，添加前缀（匹配训练时的处理）
            if add_prefix:
                text = ["aerial view of university:" + t for t in text]
            
            text_input = tokenizer(text, padding='max_length', truncation=True, 
                                 max_length=config.get('max_tokens', 50),
                                 return_tensors="pt").to(device)
            

            text_output = model.text_encoder(text_input.input_ids, 
                                           attention_mask=text_input.attention_mask, 
                                           mode='text')
            text_feat = text_output.last_hidden_state
            text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
            
            text_embeds.append(text_embed)
            text_feats.append(text_feat)
            text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    
    print(f"Text features extracted: {text_embeds.shape}")
    
    # ===== 提取图像特征 =====
    print("Extracting image features...")
    image_feats = []
    image_embeds = []
    image_ids = []
    
    with torch.no_grad():
        for images, img_names in image_loader:
            images = images.to(torch.float16).to(device)  # 匹配训练代码
            
            # 使用和训练代码完全相同的图像编码方式
            image_feat = model.vision_encoder(images)
            image_embed = model.vision_proj(image_feat[:, 0, :])
            image_embed = F.normalize(image_embed, dim=-1)

            image_feats.append(image_feat)
            image_embeds.append(image_embed)
            image_ids.extend(img_names)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)
    
    print(f"Image features extracted: {image_embeds.shape}")
    
    return text_embeds, text_feats, text_atts, image_embeds, image_feats, image_ids

def two_stage_retrieval(model, text_embeds, text_feats, text_atts, image_embeds, image_feats, 
                       image_ids, config, device):
    """两阶段检索 - 完全模仿训练代码的evaluation方法"""
    print("Performing two-stage retrieval using training code's method...")
    

    print("Stage 1: Computing similarity matrix...")
    sims_matrix = image_embeds @ text_embeds.t()
    
    # 清理内存
    del image_embeds
    del text_embeds
    gc.collect()
    

    print("Stage 2: ITM reranking...")
    num_images = len(image_ids)
    num_texts = len(text_feats)
    k_test = config.get('k_test', 256)  # 使用config中的k_test值
    
    # 创建分数矩阵 - 匹配训练代码
    score_matrix_t2i = torch.full((num_texts, num_images), -100.0, dtype=torch.float16).to(device)
    
    sims_matrix_t2i = sims_matrix.t()  # 转置以匹配训练代码
    

    for i, sims in enumerate(sims_matrix_t2i):
        if i % 100 == 0:
            print(f"Processing query {i}/{num_texts}")
        
        # 选择top-k候选图像
        topk_sim, topk_idx = sims.topk(k=min(k_test, num_images), dim=0)
        
        # 准备ITM输入 - 匹配训练代码
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        
        # 使用ITM头计算精确分数 - 匹配训练代码
        with torch.no_grad():
            output = model.text_encoder(encoder_embeds=text_feats[i].repeat(len(topk_idx), 1, 1),
                                      attention_mask=text_atts[i].repeat(len(topk_idx), 1),
                                      encoder_hidden_states=encoder_output,
                                      encoder_attention_mask=encoder_att,
                                      return_dict=True,
                                      mode='fusion')
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_t2i[i, topk_idx] = score
    
    # 转换为numpy并获取排序结果
    score_matrix_t2i_np = score_matrix_t2i.cpu().numpy()
    
    # 为每个查询获取排序后的图像ID
    final_rankings = []
    for i in range(num_texts):
        # 按分数排序获取图像索引
        sorted_indices = np.argsort(score_matrix_t2i_np[i])[::-1]  # 降序排列
        
        # 获取top-10结果
        top10_image_ids = [image_ids[idx] for idx in sorted_indices[:10]]
        final_rankings.append(top10_image_ids)
    
    print("Two-stage retrieval completed!")
    return final_rankings

def save_results(query_ids, rankings, output_file):
    """保存结果 - 修复换行和格式问题"""
    print(f"Saving results to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (query_id, top_images) in enumerate(zip(query_ids, rankings)):
            # 确保每个查询返回10个结果
            top10_images = top_images[:10] if len(top_images) >= 10 else top_images[:]
            
            # 补齐到10个结果（如果不足10个，重复最后一个）
            while len(top10_images) < 10:
                if top10_images:
                    top10_images.append(top10_images[-1])
                else:
                    top10_images.append("image_placeholder")  # 极端情况的备选
            
            # 检查图像ID格式，确保都是完整的
            valid_images = []
            for img_id in top10_images:
                if img_id and (len(img_id) == 5 or (img_id.startswith("image_") and len(img_id) > 6)):
                    valid_images.append(img_id)

                else:
                    print(f"Warning: Invalid image ID '{img_id}' for query {query_id}")
                    if valid_images:
                        valid_images.append(valid_images[-1])  # 用前一个有效ID替代
                    else:
                        valid_images.append("image_placeholder")
            
            # 格式化输出：queryID image_1 image_2 ... image_10
            result_line = f"{query_id} " + " ".join(valid_images)
            f.write(result_line)
            
            # 每行后面都加换行符（包括最后一行）
            f.write("\n")
    
    print("Results saved successfully!")
    
    # 验证输出格式
    print("Verifying output format...")
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total lines: {len(lines)}")
    if lines:
        print("Sample output format:")
        for i, line in enumerate(lines[:3]):  # 显示前3行
            parts = line.strip().split()
            if parts:
                print(f"  Line {i+1}: {parts[0]} + {len(parts)-1} images")
                print(f"    Full line: {line.strip()}")
        
        # 检查格式一致性
        all_valid = True
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 11:  # 1个query_id + 10个image_id
                print(f"Warning: Line {i+1} has {len(parts)} parts instead of 11")
                print(f"  Content: {line.strip()}")
                all_valid = False
            else:
                # 检查图像ID格式
                for j, part in enumerate(parts[1:], 1):
                    if not part.startswith('image_') or len(part) < 7:
                        print(f"Warning: Line {i+1}, part {j+1} has invalid image ID: '{part}'")
                        all_valid = False
        
        if all_valid:
            print("✓ All lines have correct format (1 query_id + 10 valid image_ids)")
        else:
            print("⚠ Some lines have incorrect format")
        
        # 显示最后几行来检查是否完整
        if len(lines) > 3:
            print("Last few lines:")
            for i, line in enumerate(lines[-2:]):
                parts = line.strip().split()
                line_num = len(lines) - 2 + i + 1
                print(f"  Line {line_num}: {parts[0] if parts else 'EMPTY'} + {len(parts)-1 if parts else 0} images")
                print(f"    Content: {line.strip()}")

def main():
    parser = argparse.ArgumentParser(description='X-VLM Two-Stage GeoText Image Inference - Training Code Compatible')
    parser.add_argument('--queries', required=True, help='Path to queries txt file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--images', required=True, help='Path to JPEG images directory')
    parser.add_argument('--output', default='submission.txt', help='Output file')
    parser.add_argument('--device', default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--batch_size', default=32, type=int, help='Image batch size')
    parser.add_argument('--add_prefix', action='store_true', help='Add "aerial view of university:" prefix to queries')
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    try:

        image_path = Path(args.images)
        if not image_path.exists():
            print(f"Error: Image directory {image_path} does not exist!")
            return
        

        model, tokenizer, config = load_xvlm_model(
            args.checkpoint, args.config, device
        )
        
        # 3. 解析查询
        queries, query_ids = parse_queries(args.queries)
        if not queries:
            print("Error: No queries loaded!")
            return
        

        image_size = config.get('image_res', 384)
        print(f"Using image size: {image_size}")
        
        image_dataset = GeoTextImageDataset(args.images, image_size)
        
        if len(image_dataset) == 0:
            print("Error: No image files found!")
            print(f"Please check if there are image files in: {args.images}")
            return
        
        image_loader = DataLoader(
            image_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=min(4, os.cpu_count()),
            pin_memory=(device == 'cuda')
        )
        

        text_embeds, text_feats, text_atts, image_embeds, image_feats, image_ids = extract_all_features(
            model, tokenizer, queries, image_loader, config, device, args.add_prefix
        )
        

        rankings = two_stage_retrieval(
            model, text_embeds, text_feats, text_atts, image_embeds, image_feats, 
            image_ids, config, device
        )
        

        save_results(query_ids, rankings, args.output)
        

        print("\nSample results (query_id + top 10 image_ids):")
        for i in range(min(3, len(query_ids))):
            query_text = queries[i][:50] + "..." if len(queries[i]) > 50 else queries[i]
            top10 = rankings[i][:10] if len(rankings[i]) >= 10 else rankings[i]
            

            while len(top10) < 10:
                if top10:
                    top10.append(top10[-1])
                else:
                    top10.append("image_placeholder")
            
            print(f"{query_ids[i]} {' '.join(top10)}")
            print(f"  Query text: '{query_text}'")
            print()
        
        print(f"\nGeoText inference completed! Results saved to {args.output}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
