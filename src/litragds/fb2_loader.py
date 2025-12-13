from bs4 import BeautifulSoup
import zipfile
import glob
import os
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FB2Loader:
    """Загрузчик и парсер файлов в формате FB2"""
    
    @staticmethod
    def parse_fb2_file(file_path: str) -> Dict[str, Any]:
        """
        Парсинг FB2 файла и извлечение структурированной информации
        """
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Проверяем, является ли файл ZIP-архивом
            if content.startswith(b'PK'):
                return FB2Loader._parse_zipped_fb2(file_path)
            else:
                return FB2Loader._parse_xml_fb2(content, file_path)
                
        except Exception as e:
            logger.error(f"Error parsing FB2 file {file_path}: {e}")
            return {}
    
    @staticmethod
    def _parse_zipped_fb2(file_path: str) -> Dict[str, Any]:
        """Парсинг запакованного FB2 файла"""
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Ищем файл с расширением .fb2 в архиве
                fb2_files = [f for f in zip_ref.namelist() if f.lower().endswith('.fb2')]
                if not fb2_files:
                    raise ValueError("No FB2 file found in archive")
                
                with zip_ref.open(fb2_files[0]) as fb2_file:
                    content = fb2_file.read()
                    return FB2Loader._parse_xml_fb2(content, file_path)
                    
        except Exception as e:
            logger.error(f"Error parsing zipped FB2 {file_path}: {e}")
            return {}
    
    @staticmethod
    def _parse_xml_fb2(content: bytes, file_path: str) -> Dict[str, Any]:
        """Парсинг XML содержимого FB2 файла"""
        try:
            # Пробуем разные кодировки
            encodings = ['utf-8', 'windows-1251', 'cp1251', 'koi8-r']
            soup = None
            
            for encoding in encodings:
                try:
                    soup = BeautifulSoup(content.decode(encoding), 'xml')
                    break
                except UnicodeDecodeError:
                    continue
            
            if soup is None:
                # Если не удалось декодировать, пробуем без указания кодировки
                soup = BeautifulSoup(content, 'xml')
            
            # Извлекаем метаданные
            metadata = FB2Loader._extract_metadata(soup, file_path)
            
            # Извлекаем содержимое
            content_sections = FB2Loader._extract_content(soup)
            
            return {
                'metadata': metadata,
                'content': content_sections,
                'file_path': file_path
            }
            
        except Exception as e:
            logger.error(f"Error parsing XML FB2 {file_path}: {e}")
            return {}
    
    @staticmethod
    def _extract_metadata(soup: BeautifulSoup, file_path: str) -> Dict[str, str]:
        """Извлечение метаданных из FB2 файла"""
        metadata = {
            'file_name': os.path.basename(file_path),
            'title': '',
            'author': '',
            'genre': '',
            'language': '',
            'year': ''
        }
        
        try:
            # Название
            title_info = soup.find('title-info')
            if title_info:
                # Заголовок
                title_tag = title_info.find('book-title')
                if title_tag:
                    metadata['title'] = title_tag.get_text(strip=True)
                
                # Автор
                author_tag = title_info.find('author')
                if author_tag:
                    first_name = author_tag.find('first-name')
                    last_name = author_tag.find('last-name')
                    author_name = ""
                    if first_name:
                        author_name += first_name.get_text(strip=True)
                    if last_name:
                        author_name += " " + last_name.get_text(strip=True)
                    metadata['author'] = author_name.strip()
                
                # Жанр
                genre_tag = title_info.find('genre')
                if genre_tag:
                    metadata['genre'] = genre_tag.get_text(strip=True)
                
                # Язык
                lang_tag = title_info.find('lang')
                if lang_tag:
                    metadata['language'] = lang_tag.get_text(strip=True)
            
            # Год издания
            publish_info = soup.find('publish-info')
            if publish_info:
                year_tag = publish_info.find('year')
                if year_tag:
                    metadata['year'] = year_tag.get_text(strip=True)
                    
        except Exception as e:
            logger.warning(f"Error extracting metadata from {file_path}: {e}")
        
        return metadata
    
    @staticmethod
    def _extract_content(soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Извлечение текстового содержимого из FB2 файла"""
        sections = []
        
        try:
            # Ищем все секции body
            bodies = soup.find_all('body')
            
            for body_idx, body in enumerate(bodies):
                body_name = body.get('name', f'body_{body_idx}')
                
                # Обрабатываем все секции внутри body
                body_sections = body.find_all('section')
                
                if not body_sections:
                    # Если нет секций, обрабатываем весь body как одну секцию
                    text_content = FB2Loader._extract_text_from_element(body)
                    if text_content:
                        sections.append({
                            'body': body_name,
                            'section_index': 0,
                            'text': text_content,
                            'title': 'Основной текст'
                        })
                else:
                    # Обрабатываем каждую секцию отдельно
                    for section_idx, section in enumerate(body_sections):
                        section_title = FB2Loader._extract_section_title(section)
                        text_content = FB2Loader._extract_text_from_element(section)
                        
                        if text_content:
                            sections.append({
                                'body': body_name,
                                'section_index': section_idx,
                                'text': text_content,
                                'title': section_title
                            })
                            
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
        
        return sections
    
    @staticmethod
    def _extract_section_title(section) -> str:
        """Извлечение заголовка секции"""
        try:
            title_tag = section.find('title')
            if title_tag:
                return title_tag.get_text(strip=True)
            
            # Ищем заголовки в параграфах
            first_p = section.find('p')
            if first_p and len(first_p.get_text(strip=True)) < 100:
                return first_p.get_text(strip=True)
                
        except:
            pass
        
        return "Без названия"
    
    @staticmethod
    def _extract_text_from_element(element) -> str:
        """Рекурсивное извлечение текста из элемента"""
        try:
            texts = []
            
            # Рекурсивно обходим все дочерние элементы
            for child in element.descendants:
                if child.name in ['p', 'v']:  # Параграфы и стихи
                    text = child.get_text(strip=True)
                    if text:
                        texts.append(text)
                elif child.name is None:  # Текстовые узлы
                    text = str(child).strip()
                    if text and not text.isspace():
                        # Проверяем, не является ли текст частью тега
                        parent = child.parent
                        if parent and parent.name not in ['title', 'subtitle', 'poem']:
                            texts.append(text)
            
            # Объединяем тексты, убирая дубликаты
            unique_texts = []
            for text in texts:
                if text and text not in unique_texts:
                    unique_texts.append(text)
            
            return '\n'.join(unique_texts)
            
        except Exception as e:
            logger.error(f"Error extracting text from element: {e}")
            return ""
    
    @staticmethod
    def load_fb2_directory(directory_path: str, file_pattern: str = "*.fb2") -> List[Dict[str, Any]]:
        """Загрузка всех FB2 файлов из директории"""
        files = glob.glob(os.path.join(directory_path, file_pattern))
        files.extend(glob.glob(os.path.join(directory_path, "*.fb2.zip")))
#        files.extend(glob.glob(os.path.join(directory_path, "*.zip")))
        
        documents = []
        
        for file_path in files:
            logger.info(f"Processing FB2 file: {file_path}")
            
            try:
                parsed_data = FB2Loader.parse_fb2_file(file_path)
                
                if parsed_data and parsed_data['content']:
                    # Создаем документы для каждой секции
                    for section in parsed_data['content']:
                        if len(section['text']) > 50:  # Минимальная длина текста
                            documents.append({
                                'text': section['text'],
                                'metadata': {
                                    'file': parsed_data['metadata']['file_name'],
                                    'title': parsed_data['metadata']['title'],
                                    'author': parsed_data['metadata']['author'],
                                    'genre': parsed_data['metadata']['genre'],
                                    'year': parsed_data['metadata']['year'],
                                    'body': section['body'],
                                    'section_title': section['title'],
                                    'section_index': section['section_index'],
                                    'language': parsed_data['metadata']['language'],
                                    'source': 'fb2'
                                }
                            })
                    
                    logger.info(f"Successfully processed {file_path}: {len(parsed_data['content'])} sections")
                else:
                    logger.warning(f"No content extracted from {file_path}")
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        
        return documents
