"""
collector.py — Data collection dari 4 platform media sosial Indonesia.

Platform coverage:
  Twitter/X   : Tweepy + Twitter API v2          | ID prefix: twitter_
  YouTube     : Google API Python Client          | ID prefix: youtube_
  TikTok      : Research API / Playwright         | ID prefix: tiktok_
  Instagram   : Instaloader                       | ID prefix: ig_

Mock data design (saat API tidak tersedia):
  - Setiap platform punya ID prefix unik
  - Setiap platform punya 20 template teks yang berbeda
  - Setiap platform punya random seed yang berbeda (11/22/33/44)
  - generate_mock() adalah method per class, bukan shared function
"""

import re
import time
import hashlib
import logging
import pandas as pd
import emoji
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


TWITTER_TEMPLATES = [
    "mantap banget sih ini 🗿🗿 gampang katanya",
    "wkwk lucu parah 😂😂😂 receh abis deh",
    "sedih banget deh 😭 kok bisa gitu sih",
    "keren sih emang 🔥🔥 respect banget lah",
    "gak nyangka beneran viral 💀 wtf banget",
    "ya ampun ini mah beneran parah 😤 kesel",
    "serius gak sih 🤔 masa iya gitu caranya",
    "gaskeun lah 💪🔥 siap jalan terus",
    "mager banget hari ini 😪 gabut poll",
    "astaga beneran gak nyambung 😑 awkward",
    "literally best thing ever 🥰✨ seneng banget",
    "oke deh terserah 👍 udah gitu aja",
    "kepo banget sama ini 👀 penasaran banget",
    "ngakak parah 😹😹 gak ketahan ketawa",
    "healing dulu yuk 🌿😌 butuh refreshing",
    "bucin parah emang 💔 gapapa sih",
    "receh bet ini kontennya 😂 tapi lucu juga",
    "gercep dong jangan lama 😤 keburu basi",
    "ya elah masa gitu 🙄 gak masuk akal",
    "mantap jiwa 🫡 respect tinggi buat ini",
]

YOUTUBE_TEMPLATES = [
    "konten ini bagus banget sih 👍👍 subscribe dulu ah",
    "wkwk ngakak parah nonton ini 😂😂 lucu bet",
    "keren banget editannya 🔥✨ respect sama creator",
    "relate banget sama situasi ini 😭😭 real talk",
    "gilak beneran gak nyangka plot twistnya 💀 anjir",
    "tutorial ini helpful banget 🙏✨ makasih banyak",
    "gak nyangka bisa sebagus ini 😍 submas dulu ah",
    "receh tapi menghibur 😄 terus berkarya ya kak",
    "serius bagus banget content quality nya 🎬🔥 top",
    "ini mah beneran bikin nangis 😢💔 sedih parah",
    "sumpah lucu banget 😂🤣 nonton berkali kali",
    "mantap bet ini video 💯🔥 gas terus kontennya",
    "akhirnya ada yang bahas ini 👏👏 nunggu lama",
    "beneran detail banget penjelasannya 📚✨ keren",
    "gak bosen nonton ini 🎵😍 enak banget emang",
    "ya ampun baru tau ada yang kayak gini 😲👀",
    "edit video ini smooth banget 🎬✨ aesthetic parah",
    "haha parah ini 😂😂😂 lawak banget sih",
    "informasinya bermanfaat banget 🙏📖 makasih kak",
    "pengen coba juga nih 🤩💪 inspiratif banget",
]

TIKTOK_TEMPLATES = [
    "mantap banget sih 🗿 gampang banget katanya",
    "wkwkwk 💀💀 ga ada obatnya emang",
    "fyp dong please 🙏🙏 udah ngebantu spread",
    "ini aku banget 😭✋ relate parah sumpah",
    "gak ada yang nanya tapi aku jawab 🗿",
    "beneran gak ketawa 💀 serius ini",
    "skill issue 🗿 maaf ya",
    "ngapain sih 😭😭 ga ada kerjaan",
    "literally me everyday 💀😭",
    "ratio plus L plus skill issue 🗿",
    "oke sip mantap 🫡 lanjut",
    "ga ada yang minta pendapat lo 🗿",
    "aduh kenapa gini sih 😭🙏 capek",
    "ini real atau editan 👀👀 penasaran",
    "kenapaaaa 😭💀 aku mau nangis",
    "itu aku yang komentar 🫵😂 ketahuan",
    "lanjutkan kak 🔥🔥 bikin konten terus",
    "aku gak ngerti 🤡 mungkin aku yang bego",
    "plot twist terkejut 😲💀 ga nyangka",
    "receh tapi bener juga sih 🗿 mikir",
]

INSTAGRAM_TEMPLATES = [
    "cantik banget 😍😍 goals bgt sih aesthetic",
    "outfit kece banget 🔥 mau dong info brandnya",
    "ih lucu banget 🥰✨ gemoy sumpah",
    "ini tempat makan dimana? 🤤 pengen kesana",
    "serius ini enak banget 😭😋 pengen nyoba",
    "foto bagus banget editannya 📸✨ preset apa ini?",
    "glow up parah 😍🌟 cantik banget sekarang",
    "tempat ini aesthetic banget 🌸📍 mau kesini",
    "ootd keren banget 👗🔥 inspirasi banget sih",
    "makanan ini enak banget kayaknya 🍜😍 wajib coba",
    "view nya indah banget 🌅✨ healing spot banget",
    "beneran bagus banget hasilnya 💄😍 skincare apa?",
    "caption ini relate banget 💕🥺 touching banget",
    "ini beneran natural? 😲✨ glowing banget",
    "koleksinya lucu lucu 🛍️😍 mau semua",
    "feed nya aesthetic banget 🎨✨ konsisten",
    "resepnya boleh dong share 🍳😋 keliatan enak",
    "traveling goals banget 🌏✈️ pengen kesini",
    "couple goals banget 💑❤️ sweet parah",
    "hair goals banget 💇✨ salon mana ini?",
]


def _build_dataframe(records, platform, id_prefix):
    for i, rec in enumerate(records):
        rec["id"]       = f"{id_prefix}_{i:06d}"
        rec["platform"] = platform
    return pd.DataFrame(records)


class InclusionFilter:
    def __init__(self, min_emoji_count=1, min_token_count=3):
        self.min_emoji_count = min_emoji_count
        self.min_token_count = min_token_count
        self._pat = re.compile(
            "(" + "|".join(
                re.escape(e)
                for e in sorted(emoji.EMOJI_DATA.keys(), key=len, reverse=True)
            ) + ")"
        )

    def passes(self, text):
        if not text or not isinstance(text, str):
            return False
        if len(self._pat.findall(text)) < self.min_emoji_count:
            return False
        cleaned = self._pat.sub("", text).strip()
        return len(cleaned.split()) >= self.min_token_count


class DuplicateFilter:
    def __init__(self):
        self._seen = set()

    def is_duplicate(self, text):
        h = hashlib.md5(text.strip().lower().encode()).hexdigest()
        if h in self._seen:
            return True
        self._seen.add(h)
        return False

    def reset(self):
        self._seen.clear()


class TwitterCollector:
    def __init__(self, bearer_token):
        import tweepy
        self.client           = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
        self.inclusion_filter = InclusionFilter()
        self.dedup_filter     = DuplicateFilter()

    def collect(self, queries, target_count=20000, max_results=100):
        import tweepy
        records = []
        for query in queries:
            if len(records) >= target_count:
                break
            full_query = f"{query} lang:id -is:retweet -is:reply"
            try:
                paginator = tweepy.Paginator(
                    self.client.search_recent_tweets,
                    query=full_query,
                    tweet_fields=["id", "text", "created_at"],
                    max_results=max_results,
                    limit=(target_count // max_results) + 10,
                )
                for response in paginator:
                    if not response.data:
                        continue
                    for tweet in response.data:
                        text = tweet.text
                        if self.dedup_filter.is_duplicate(text):
                            continue
                        if not self.inclusion_filter.passes(text):
                            continue
                        records.append({
                            "text"     : text,
                            "timestamp": tweet.created_at.isoformat() if tweet.created_at else None,
                        })
                        if len(records) >= target_count:
                            break
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"Twitter error: {e}")
                continue
        return _build_dataframe(records, "twitter", "twitter")

    def generate_mock(self, count):
        import random
        random.seed(11)
        records = [
            {"text": random.choice(TWITTER_TEMPLATES), "timestamp": datetime.now(timezone.utc).isoformat()}
            for _ in range(count)
        ]
        return _build_dataframe(records, "twitter", "twitter")


class YouTubeCollector:
    TARGET_KEYWORDS = [
        "gaming indonesia", "meme indonesia", "drama korea reaction",
        "review hp indonesia", "viral tiktok indonesia",
        "mukbang indonesia", "review makanan indonesia",
    ]

    def __init__(self, api_key):
        from googleapiclient.discovery import build
        self.youtube          = build("youtube", "v3", developerKey=api_key)
        self.inclusion_filter = InclusionFilter()
        self.dedup_filter     = DuplicateFilter()

    def collect(self, target_count=12500):
        records = []
        for keyword in self.TARGET_KEYWORDS:
            if len(records) >= target_count:
                break
            for vid_id in self._search_video_ids(keyword):
                if len(records) >= target_count:
                    break
                records.extend(self._collect_comments(vid_id))
                time.sleep(1.0)
        return _build_dataframe(records[:target_count], "youtube", "youtube")

    def _search_video_ids(self, query, max_results=20):
        try:
            response = self.youtube.search().list(
                q=query, part="id", type="video",
                relevanceLanguage="id", regionCode="ID",
                maxResults=max_results,
            ).execute()
            return [item["id"]["videoId"] for item in response.get("items", [])]
        except Exception as e:
            logger.error(f"YouTube search error: {e}")
            return []

    def _collect_comments(self, video_id, max_comments=300):
        comments, page_token = [], None
        while len(comments) < max_comments:
            try:
                kwargs = dict(part="snippet", videoId=video_id,
                              maxResults=min(100, max_comments - len(comments)),
                              textFormat="plainText")
                if page_token:
                    kwargs["pageToken"] = page_token
                response = self.youtube.commentThreads().list(**kwargs).execute()
                for item in response.get("items", []):
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    text    = snippet.get("textOriginal", "")
                    if not text or self.dedup_filter.is_duplicate(text):
                        continue
                    if not self.inclusion_filter.passes(text):
                        continue
                    comments.append({"text": text, "timestamp": snippet.get("publishedAt")})
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
                time.sleep(0.3)
            except Exception as e:
                logger.error(f"YouTube comment error: {e}")
                break
        return comments

    def generate_mock(self, count):
        import random
        random.seed(22)
        records = [
            {"text": random.choice(YOUTUBE_TEMPLATES), "timestamp": datetime.now(timezone.utc).isoformat()}
            for _ in range(count)
        ]
        return _build_dataframe(records, "youtube", "youtube")


class TikTokCollector:
    def __init__(self):
        self.inclusion_filter = InclusionFilter()
        self.dedup_filter     = DuplicateFilter()

    def collect(self, target_count=10000, use_mock=False):
        if use_mock:
            return self.generate_mock(target_count)
        logger.warning("TikTok Playwright requires manual session. Use use_mock=True.")
        return pd.DataFrame()

    def generate_mock(self, count):
        import random
        random.seed(33)
        records = [
            {"text": random.choice(TIKTOK_TEMPLATES), "timestamp": datetime.now(timezone.utc).isoformat()}
            for _ in range(count)
        ]
        return _build_dataframe(records, "tiktok", "tiktok")


class InstagramCollector:
    def __init__(self, username=None, password=None):
        self.inclusion_filter = InclusionFilter()
        self.dedup_filter     = DuplicateFilter()
        self._username        = username
        self._password        = password

    def collect(self, target_accounts=None, target_count=7500, use_mock=False):
        if use_mock or not (self._username and self._password):
            return self.generate_mock(target_count)
        import instaloader
        loader, records = instaloader.Instaloader(), []
        try:
            loader.login(self._username, self._password)
        except Exception as e:
            logger.warning(f"Instagram login failed: {e}. Using mock.")
            return self.generate_mock(target_count)
        for account in (target_accounts or []):
            if len(records) >= target_count:
                break
            try:
                profile = instaloader.Profile.from_username(loader.context, account)
                for post in profile.get_posts():
                    if len(records) >= target_count:
                        break
                    for comment in post.get_comments():
                        text = comment.text
                        if self.dedup_filter.is_duplicate(text):
                            continue
                        if not self.inclusion_filter.passes(text):
                            continue
                        records.append({"text": text, "timestamp": comment.created_at_utc.isoformat()})
                    time.sleep(2.0)
            except Exception as e:
                logger.error(f"Instagram error ({account}): {e}")
        if not records:
            return self.generate_mock(target_count)
        return _build_dataframe(records[:target_count], "instagram", "ig")

    def generate_mock(self, count):
        import random
        random.seed(44)
        records = [
            {"text": random.choice(INSTAGRAM_TEMPLATES), "timestamp": datetime.now(timezone.utc).isoformat()}
            for _ in range(count)
        ]
        return _build_dataframe(records, "instagram", "ig")
